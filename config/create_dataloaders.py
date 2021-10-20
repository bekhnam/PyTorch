from . import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os

def get_dataloader(rootDir, transforms, bs, shuffle=True):
    ds = datasets.ImageFolder(root=rootDir, transform=transforms)
    loader = DataLoader(ds, batch_size=bs, shuffle=shuffle,
                        num_workers=4,
                        pin_memory=True if config.DEVICE == "cuda" else False)

    return (ds, loader)
