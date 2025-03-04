import pytorch_lightning as pl
import torch
from lr_schedulers import WarmupCosineAnnealingScheduler
    
class PhotozLightning(pl.LightningModule):
    """
    A PyTorch Lightning module for photometric redshift estimation.

    Args:
        encoder (nn.Module): A CNN encoder.
        encoder_mlp (nn.Module, optional): An optional MLP to further process encoder outputs and reduce dimensionality.
        photoz_mlp (nn.Module): The final MLP for redshift prediction.
        lr (float): Learning rate for the optimizer.
        transforms (callable): Optional image transformations (e.g., augmentations).  
    """
    
    def __init__(self, encoder: torch.nn.Module=None, encoder_mlp: torch.nn.Module=None,
                 photoz_mlp: torch.nn.Module=None, lr: float=None, transforms=None):
        super().__init__()
        self.encoder = encoder
        self.encoder_mlp = encoder_mlp
        self.photoz_mlp = photoz_mlp
        self.lr = lr
        self.transforms = transforms
        
    def forward(self, x):
        """
        Forward pass through the encoder, optional MLP, and the final MLP.
        """
        x = self.encoder(x)
        if self.encoder_mlp is not None:
            x = self.encoder_mlp(x)
        x = self.photoz_mlp(x)
        return x
    
    def redshift_loss(self, pred_redshifts, true_redshifts):
        """
        Huber loss with delta=0.15.
        """
        loss = torch.nn.HuberLoss(delta=0.15)
        return loss(pred_redshifts, true_redshifts)
    
    def training_step(self, batch_data, batch_idx):
        """
        Training step: processes the batch, computes the loss, and logs metrics.
        """
        batch_images, batch_redshifts, batch_weights, _ = batch_data
        
        if self.transforms is not None:
            batch_redshift_predictions = self.forward(self.transforms(batch_images)).squeeze()
        else:
            batch_redshift_predictions = self.forward(batch_images).squeeze()
        
        # assert Pytorch output and true redshifts/weights have same shape
        assert batch_redshifts.shape == batch_redshift_predictions.shape
        assert batch_redshift_predictions.shape == batch_weights.shape
        
        loss = self.redshift_loss(batch_redshift_predictions, batch_redshifts)
        self.log("training_loss", loss, on_epoch=True, sync_dist=True)
        
        # Compute metrics (bias, NMAD, and outlier fraction)
        delta = (batch_redshift_predictions - batch_redshifts) / (1+batch_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826*torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15)/len(batch_redshifts)
        
        # Log the metrics with tensorboard
        self.log('training_bias', bias, on_epoch=True, sync_dist=True)
        self.log('training_nmad', nmad, on_epoch=True, sync_dist=True)
        self.log('training_outlier_f', outlier_fraction, on_epoch=True, sync_dist=True)
        
        return loss
        
    def validation_step(self, batch_data, batch_idx):
        """
        Same as training step but for validation data.
        """
        batch_images, batch_redshifts, batch_weights, _ = batch_data
        
        if self.transforms is not None:
            batch_redshift_predictions = self.forward(self.transforms(batch_images)).squeeze()
        else:
            batch_redshift_predictions = self.forward(batch_images).squeeze()

        # assert Pytorch output and true redshifts/weights have same shape
        assert batch_redshifts.shape == batch_redshift_predictions.shape
        assert batch_redshift_predictions.shape == batch_weights.shape
        
        loss = self.redshift_loss(batch_redshift_predictions, batch_redshifts)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        
        delta = (batch_redshift_predictions - batch_redshifts) / (1+batch_redshifts)
        bias = torch.mean(delta)
        nmad = 1.4826*torch.median(torch.abs(delta-torch.median(delta)))
        outlier_fraction = torch.sum(torch.abs(delta)>0.15)/len(batch_redshifts)
        
        self.log('val_bias', bias, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_nmad', nmad, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_outlier_f', outlier_fraction, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configures the optimizer (Adam) and the learning rate scheduler.
        """
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[150,700,900], gamma=0.1)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=1, gamma=0.1)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=900, eta_min=5e-7)
        lr_scheduler = WarmupCosineAnnealingScheduler(
            optimizer=optim,
            warmup_epochs=100,
            cos_half_period=500,
            min_lr=5e-6
        )
        
        return [optim], [lr_scheduler]