import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class sim_clr_lightning(pl.LightningModule):
    def __init__(self, encoder, encoder_mlp, projection_head, transforms, lr, loss_type='contrastive'):
        super().__init__()
        self.encoder = encoder
        self.encoder_mlp = encoder_mlp
        self.projection_head = projection_head
        self.transforms = transforms
        self.lr = lr
        self.loss_type = loss_type
        
    def forward(self, x):
        x = self.encoder(x)
        # in case projection head from CNN output is separated into two mlps, an encoder_mlp and a projection head
        if self.encoder_mlp is not None:
            x = self.encoder_mlp(x)
        x = self.projection_head(x)
        return x
    
    def contrastive_loss(self, projections, temp=0.07):
        # first half of projections and second half of projections are positive pairs
        cos_sim = F.cosine_similarity(projections[:,None,:], projections[None,:,:], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -1e4)
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
        cos_sim = cos_sim / temp

        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)

        return nll.mean()
    
    def calculate_unif_and_align(self, projections):
        first_batch_projections = projections[:projections.shape[0]//2,:]
        positive_pair_projections = projections[projections.shape[0]//2:,:]
        l_align = (first_batch_projections - positive_pair_projections).norm(dim=1).pow(2).mean()
        sq_pdist1 = torch.pdist(first_batch_projections, p=2).pow(2)
        l_unif1 = sq_pdist1.mul(-2).exp().mean().log()
        sq_pdist2 = torch.pdist(positive_pair_projections, p=2).pow(2)
        l_unif2 = sq_pdist2.mul(-2).exp().mean().log()
        
        return l_align, (l_unif1 + l_unif2)/2
    
    def align_unif_loss(self, projections, lamda=1):
        # this is for using alignment + uniformity loss from Wang & Isola 2022
        l_align, l_unif = self.calculate_unif_and_align(projections)
        
        return l_align + lamda*l_unif
    
    def training_step(self, batch_images, batch_idx):
        # transform batch images twice to create batch_size positive pairs (batch_size*2 images in total)
        batch_transformed_images = torch.cat((self.transforms(batch_images), self.transforms(batch_images)), dim=0)
        projections = self.forward(batch_transformed_images)
        if self.loss_type == 'contrastive':
            loss = self.contrastive_loss(projections)
        elif self.loss_type == 'align_unif':
            loss = self.align_unif_loss(projections, lamda=0.5)
        self.log("training_loss", loss, on_epoch=True, sync_dist=True)
        
        l_align, l_unif = self.calculate_unif_and_align(projections)
        self.log("l_align_train", l_align, on_epoch=True, sync_dist=True)
        self.log("l_unif_train", l_unif, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch_images, batch_idx):
        batch_transformed_images = torch.cat((self.transforms(batch_images), self.transforms(batch_images)), dim=0)
        projections = self.forward(batch_transformed_images)
        if self.loss_type == 'contrastive':
            loss = self.contrastive_loss(projections)
        elif self.loss_type == 'align_unif':
            loss = self.align_unif_loss(projections, lamda=0.5)
        self.log("validation_loss", loss, on_epoch=True, sync_dist=True)
        
        l_align, l_unif = self.calculate_unif_and_align(projections)
        self.log("l_align_train", l_align, on_epoch=True, sync_dist=True)
        self.log("l_unif_train", l_unif, on_epoch=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=[4000], gamma=0.1)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=1, gamma=0.1)
        #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=5, eta_min=1e-5)

        return [optim], [lr_scheduler]
        #return optim