import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.transforms import v2
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch import loggers as pl_loggers

# import custom modules
import photoz_lightning as photoz_lightning
import nn_modules_pytorch as nn_modules
import candels_data_modules as dm
import sdss_reddening as reddening
import transform_utils as transform_utils

if __name__ == '__main__':
    lr = 5e-4
    n_filters = 4
    batch_size = 128
    im_id = 2 # 0 is for arcsinh # 2 is no non-linear transformation

    files = ['candels_train_catalog.hdf5', 'candels_val_catalog.hdf5',
             'candels_test_catalog.hdf5'] 
    path_to_file = '/pscratch/sd/a/ashodkh/image_photo_z/' + files[0]
    path_to_val = '/pscratch/sd/a/ashodkh/image_photo_z/' + files[1]
    
    path_to_quantiles = '/global/homes/a/ashodkh/image_photo_z/notebooks/candels_train_image_quantiles.npy'
    reddening = reddening.candels_reddening(deredden=True)
    transforms = v2.Compose([
        v2.RandomHorizontalFlip(0.5),
        v2.RandomRotation(180, interpolation=v2.InterpolationMode.BILINEAR),
        transform_utils.JitterCrop(output_dim=64, jitter_lim=8),
    ])
    #transforms = transform_utils.JitterCrop(output_dim=64, jitter_lim=None)
    t_id = 0 # 0 is Hayat, 1 is only Crop
    data_module = dm.candels_images_data_module(batch_size=batch_size, num_workers=28, train_size=0.8,
                                             path_to_file=path_to_file, path_to_quantiles=path_to_quantiles,
                                             path_to_val=path_to_val,
                                             with_labels=True, transforms=reddening, load_ebv=True,
                                             im_id=im_id)
    
    #encoder_type = 'convnext'    
    #encoder = models.convnext_tiny(weights=None)
    #encoder._modules["features"][0][0] = nn.Conv2d(5, 96, kernel_size=(4,4), stride=(4,4))
    encoder_type = 'my_encoder'
    joint_blocks = nn_modules.JointBlocks(input_channels=32, block_channels=[32,64], avg_pooling_layers=[4,4])
    encoder = nn_modules.Encoder(input_channels=n_filters, first_layer_output_channels=32, joint_blocks=joint_blocks)
    redshift_mlp = nn_modules.MLP(input_dim=1024, hidden_layers=[512,256,64,1])  
    
    photoz_model = photoz_lightning.PhotozLightning(
        encoder=encoder,
        encoder_mlp=None,
        photoz_mlp=redshift_mlp,
        lr=lr,
        transforms=transforms
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='/pscratch/sd/a/ashodkh/image_photo_z/model_weights/',
        filename=f'candels_{n_filters}photoz_supervised_{encoder_type}_im_id{im_id}_t{t_id}_lr{lr}_' + '{epoch}',
        every_n_epochs=50,
        save_top_k=-1,
        enable_version_counter=False
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="/global/homes/a/ashodkh/image_photo_z/scripts/lightning_logs_candels",
        name=f'candels_{n_filters}photoz_supervised_{encoder_type}_im_id{im_id}_t{t_id}_lr{lr}'
    )
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=4,
        min_epochs=1,
        max_epochs=10000,
        precision='16-mixed',
        log_every_n_steps=1,
        default_root_dir="/global/homes/a/ashodkh/image_photo_z/scripts",
        strategy='ddp', logger=tb_logger, profiler="simple",
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, lr_monitor_callback]
    )
    trainer.fit(photoz_model, data_module)