import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch




class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=1, data_root="./data", train_split_ratio=0.8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.train_split_ratio = train_split_ratio

    def prepare_data(self):
        # Download the dataset if needed (only using rank 1)
        datasets.CIFAR100(root=self.data_root, train=True, download=True)
    

    def setup(self, stage=None):
        # Split the dataset into train and validation sets
        train_dataset = datasets.CIFAR100(root=self.data_root, train=True, transform=self.get_transforms("train"))
        val_dataset = datasets.CIFAR100(root=self.data_root, train=True, transform=self.get_transforms("val"))
        indices = torch.randperm(len(train_dataset))
        val_size = int(len(train_dataset) * self.train_split_ratio)
        self.train_dataset = Subset(train_dataset, indices[-val_size:])
        self.val_dataset = Subset(val_dataset, indices[:-val_size])
       
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size * 2, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        test_dataset = datasets.CIFAR100(root=self.data_root, train=False, transform=self.get_transforms("test"))
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True, shuffle=False)
    
    def get_transforms(self,split):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        shared_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std) 
        ]
        
        if split == "train":
            return transforms.Compose([
                *shared_transforms,
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                transforms.RandomHorizontalFlip(),
                # ...
            ])
            
        elif split == "val":
            return transforms.Compose([
                *shared_transforms,
                # ...
            ])
        elif split == "test":
            return transforms.Compose([
                *shared_transforms,
                # ...
            ])