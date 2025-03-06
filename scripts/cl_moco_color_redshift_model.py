import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lr_schedulers import WarmupCosineAnnealingScheduler
import copy

class SimCLRMoCoLightning(pl.LightningModule):
    def __init__(
        self,
        encoder=None,
        encoder_mlp=None,
        projection_head=None,
        redshift_mlp=None,
        color_mlp=None,
        transforms=None,
        lr=None,
        loss_type='contrastive',
        momentum=0.999,
        queue_size=50000,
        temperature=0.1
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_mlp = encoder_mlp
        self.projection_head = projection_head
        self.redshift_mlp = redshift_mlp
        self.color_mlp = color_mlp
        self.transforms = transforms
        self.lr = lr
        self.loss_type = loss_type
        self.momentum = momentum
        self.temperature = temperature
        #self.loss_weights = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0]), requires_grad=True)
        
        # Initialize the momentum (key) encoder and its heads as copies of the original
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.momentum_encoder_mlp = copy.deepcopy(self.encoder_mlp) if encoder_mlp else None
        self.momentum_projection_head = copy.deepcopy(self.projection_head)
        
        # Freeze all parameters in the momentum encoder and its heads
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
        if self.momentum_encoder_mlp:
            for param in self.momentum_encoder_mlp.parameters():
                param.requires_grad = False
        for param in self.momentum_projection_head.parameters():
            param.requires_grad = False
        
        # Initialize the queue for negative samples
        self.register_buffer("queue", torch.randn(queue_size, projection_head.out_features))
        self.queue = F.normalize(self.queue, dim=1)  # normalize
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def forward(self, x, use_momentum_encoder=False):
        """Forward pass through the encoder, MLP, and projection head."""
        if use_momentum_encoder:
            x = self.momentum_encoder(x)
            if self.momentum_encoder_mlp:
                x = self.momentum_encoder_mlp(x)
            x_proj = self.momentum_projection_head(x)
        else:
            x = self.encoder(x)
            if self.encoder_mlp:
                x = self.encoder_mlp(x)
            x_proj = self.projection_head(x)
        
        if self.redshift_mlp is not None:
            x_redshift = self.redshift_mlp(x)
        else:
            x_redshift = None
        x_color = self.color_mlp(x)
        
        return F.normalize(x_proj, dim=1), x_redshift.squeeze(), x_color
    
    @torch.no_grad()
    def update_momentum_encoder(self):
        """Update momentum encoder and its heads using exponential moving average (EMA)."""
        for param_q, param_k in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
        if self.encoder_mlp and self.momentum_encoder_mlp:
            for param_q, param_k in zip(self.encoder_mlp.parameters(), self.momentum_encoder_mlp.parameters()):
                param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
        for param_q, param_k in zip(self.projection_head.parameters(), self.momentum_projection_head.parameters()):
            param_k.data = self.momentum * param_k.data + (1.0 - self.momentum) * param_q.data
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        # Gather keys from all GPUs
        gathered_keys = [torch.zeros_like(keys) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_keys, keys)
        gathered_keys = torch.cat(gathered_keys, dim=0)  # Concatenate all keys

        # Use the gathered keys to update the queue (same on each GPU)
        batch_size = gathered_keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if (self.queue.shape[0] - ptr) < gathered_keys.shape[0]:
            self.queue[ptr:ptr + batch_size, :] = gathered_keys[0:int(self.queue.shape[0]-ptr), :]
            ptr = 0
        else:
            self.queue[ptr:ptr + batch_size, :] = gathered_keys
            ptr = (ptr + batch_size) % self.queue.shape[0]
        self.queue_ptr[0] = ptr
    
    def contrastive_loss(self, queries, keys):
        """Compute contrastive loss for MoCo using a memory queue of negative samples."""
        # Positive logits: Nx1 (dot product of each query with its corresponding key)
        pos_logits = torch.einsum('nc,nc->n', [queries, keys]).unsqueeze(-1) / self.temperature
        
        # Negative logits: NxK (dot product of each query with all keys in the queue)
        neg_logits = torch.einsum('nc,kc->nk', [queries, self.queue.clone().detach()]) / self.temperature
        
        # Combine positive and negative logits
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Cross entropy loss to maximize similarity with positive keys and dissimilarity with negatives
        return torch.mean(pos_logits)*self.temperature, F.cross_entropy(logits, labels)
    
    def weighted_mse_loss(self, predictions, truths, weights):
        """
        A weighted mse loss
        """
        mse_loss = torch.mean((predictions - truths) ** 2 * weights)
        return mse_loss
    
    def redshift_loss(self, pred_redshifts, true_redshifts):
        """
        Huber loss with delta=0.15.
        """
        loss = torch.nn.HuberLoss(delta=0.15)
        return loss(pred_redshifts, true_redshifts)
    
    def training_step(self, batch_data, batch_idx):
        batch_images, batch_redshifts, batch_redshift_weights, batch_colors = batch_data
        # Apply transformations to create two augmented views
        view_1 = self.transforms(batch_images)
        view_2 = self.transforms(batch_images)
        
        # Forward pass for query (main encoder) and key (momentum encoder)
        queries, redshift_predictions, color_predictions = self.forward(view_1) # Queries, redshifts, and colors from main encoder
        
        with torch.no_grad():  # No gradients for momentum encoder
            keys, _, _ = self.forward(view_2, use_momentum_encoder=True)  # Keys from momentum encoder

        # Update the momentum encoder and enqueue the keys
        self.update_momentum_encoder()
        self.dequeue_and_enqueue(keys)
        
        # Compute and log the contrastive loss
        pos_sim, cl_loss = self.contrastive_loss(queries, keys)
        cl_loss = cl_loss/400
        
        self.log("cl_training_loss", cl_loss, on_epoch=True, sync_dist=True)
        self.log("training_pos_sim", pos_sim, on_epoch=True, sync_dist=True)
        
        good_redshifts = batch_redshift_weights == 1
        if good_redshifts is None:
            redshift_loss = 0
        else:
            redshift_predictions = redshift_predictions[good_redshifts]
            batch_redshifts = batch_redshifts[good_redshifts]
            batch_redshift_weights = batch_redshift_weights[good_redshifts]
            redshift_loss = self.redshift_loss(redshift_predictions, batch_redshifts)
            redshift_loss = 10 * redshift_loss
        self.log("redshift_training_loss", redshift_loss, on_epoch=True, sync_dist=True)
        
        delta = (redshift_predictions - batch_redshifts) / (1+batch_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826*torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15)/len(batch_redshifts)
        
        self.log('training_bias', bias, on_step=True, on_epoch=True, sync_dist=True)
        self.log('training_nmad', nmad, on_step=True, on_epoch=True, sync_dist=True)
        self.log('training_outlier_f', outlier_fraction, on_step=True, on_epoch=True, sync_dist=True)
        
        color_loss = self.weighted_mse_loss(color_predictions, batch_colors, 1)
        color_loss = 10 * color_loss
        self.log("color_training_loss", color_loss, on_epoch=True, sync_dist=True)
        
        total_loss = cl_loss + color_loss + redshift_loss
        self.log("total_training_loss", total_loss, on_epoch=True, sync_dist=True)

        return total_loss
    
    def validation_step(self, batch_data, batch_idx):
        batch_images, batch_redshifts, batch_redshift_weights, batch_colors = batch_data
        # Apply transformations to create two augmented views
        view_1 = self.transforms(batch_images)
        view_2 = self.transforms(batch_images)
        
        # Forward pass for query (main encoder) and key (momentum encoder)
        queries, redshift_predictions, color_predictions = self.forward(view_1) # Queries, redshifts, and colors from main encoder
        with torch.no_grad():  # No gradients for momentum encoder
            keys, _, _ = self.forward(view_2, use_momentum_encoder=True)  # Keys from momentum encoder

        # Update the momentum encoder and enqueue the keys
        self.update_momentum_encoder()
        self.dequeue_and_enqueue(keys)
        
        # Compute and log the contrastive loss
        pos_sim, cl_loss = self.contrastive_loss(queries, keys)
        cl_loss = cl_loss/400
        self.log("cl_validation_loss", cl_loss, on_epoch=True, sync_dist=True)
        self.log("validation_pos_sim", pos_sim, on_epoch=True, sync_dist=True)
        
        good_redshifts = batch_redshift_weights == 1
        redshift_predictions = redshift_predictions[good_redshifts]
        batch_redshifts = batch_redshifts[good_redshifts]
        batch_redshift_weights = batch_redshift_weights[good_redshifts]
        redshift_loss = self.redshift_loss(redshift_predictions, batch_redshifts)
        redshift_loss = 10 * redshift_loss
        self.log("redshift_validation_loss", redshift_loss, on_epoch=True, sync_dist=True)

        delta = (redshift_predictions - batch_redshifts) / (1+batch_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826*torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15)/len(batch_redshifts)
        
        self.log('val_bias', bias, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_nmad', nmad, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_outlier_f', outlier_fraction, on_step=True, on_epoch=True, sync_dist=True)
        
        color_loss = self.weighted_mse_loss(color_predictions, batch_colors, 1)
        color_loss = 10 * color_loss
        self.log("color_validation_loss", color_loss, on_epoch=True, sync_dist=True)
        
        total_loss = cl_loss + color_loss + redshift_loss
        self.log("total_validation_loss", total_loss, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)

        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[10000], gamma=0.1)

        # lr_scheduler = WarmupCosineAnnealingScheduler(
        #     optimizer=optim,
        #     warmup_epochs=10,
        #     cos_half_period=1000,
        #     min_lr=5e-6
        # )
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=500, eta_min=1e-5)
        
        return [optim], [lr_scheduler]
        