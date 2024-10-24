import pytorch_lightning as pl
import torch
import h5py

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_file: str = None,
        with_labels: bool = False,
        with_weights: bool = False,
        reddening_transform = None,
        load_ebv: bool = False,
        im_id: int = 2,
    ):
        super().__init__()
        self.h5_file = h5py.File(path_to_file, 'r')
        self.with_labels = with_labels
        self.with_weights = with_weights
        self.reddening_transform = reddening_transform
        self.load_ebv = load_ebv
        self.im_id = im_id
        
    def __len__(self) -> int:
        return len(self.h5_file['images'])
    
    def __getitem__(self, idx: int):
        image = self.h5_file['images'][idx]
        ebv = self.h5_file['ebvs'][idx] if self.load_ebv else None
        redshift = self.h5_file['redshifts'][idx] if self.with_labels else None
        weight = self.h5_file['redshift_weights'][idx] if self.with_weights else 1

        # Apply reddening transformation if provided
        if self.reddening_transform:
            image = self.reddening_transform([image, ebv])

        if self.with_labels:
            return image, redshift, weight
        else:
            return image

        
class candels_images_data_module(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 1,
        train_size: float = 0.8,
        path_to_file: str = None,
        path_to_val: str = None,
        with_labels: bool = False,
        with_weights: bool = False,
        reddening_transform = None,
        load_ebv: bool = False,
        im_id: int = 2,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.path_to_file = path_to_file
        self.path_to_val = path_to_val
        self.with_labels = with_labels
        self.with_weights = with_weights
        self.reddening_transform = reddening_transform
        self.load_ebv = load_ebv
        self.im_id = im_id
        
    def setup(self, stage):
        if self.path_to_val:
            self.images_train = self._create_dataset(self.path_to_file)
            self.images_val = self._create_dataset(self.path_to_val)
        else:
            full_dataset = self._create_dataset(self.path_to_file)
            generator = torch.Generator().manual_seed(42)
            self.images_train, self.images_val = torch.utils.data.random_split(
                full_dataset,
                [self.train_size, 1 - self.train_size],
                generator=generator
            )
        
    def _create_dataset(self, path: str) -> ImagesDataset:
        """Helper to create a dataset with consistent parameters."""
        return ImagesDataset(
            path_to_file=path,
            with_labels=self.with_labels,
            with_weights=self.with_weights,
            reddening_transform=self.reddening_transform,
            load_ebv=self.load_ebv,
            im_id=self.im_id,
        )
     
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the training dataloader."""
        return self._create_dataloader(self.images_train, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the validation dataloader."""
        return self._create_dataloader(self.images_val, shuffle=False)

    def _create_dataloader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.DataLoader:
        """Helper to create a DataLoader with consistent parameters."""
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=True,
            pin_memory=torch.cuda.is_available(),
        )