"""Dataloader utilities for training, validation, and testing."""

import os
from argparse import Namespace

from PIL import Image

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset



class TestDataset(Dataset):
    """Dataset class for loading test images without labels."""

    def __init__(self, root: str, transform=None):
        """
        Args:
            root (str): Path to the test image folder.
            transform: torchvision transform to apply.
        """
        self.root = root
        self.transform = transform
        self.filename = [os.path.basename(f) for f in os.listdir(self.root)]

    def __len__(self) -> int:
        return len(self.filename)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.root, self.filename[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.filename[idx]


def dataloader(args: Namespace, mode: str) -> DataLoader:
    """
    Create dataloader based on the mode: train, val, or test.

    Args:
        args (Namespace): Command-line arguments containing data_path and batch_size.
        mode (str): Mode of the data loader ('train', 'val', 'test').

    Returns:
        DataLoader: PyTorch DataLoader for the corresponding dataset.
    """

    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomChoice([
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2)
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = None
    shuffle = False

    if mode == 'train':
        path = os.path.join(args.data_path, 'train')
        dataset = ImageFolder(root=path, transform=train_transform)
        shuffle = True
    elif mode == 'val':
        path = os.path.join(args.data_path, 'val')
        dataset = ImageFolder(root=path, transform=transform)
    elif mode == 'test':
        path = os.path.join(args.data_path, 'test')
        dataset = TestDataset(root=path, transform=transform)
    else:
        raise ValueError("Mode should be one of ['train', 'val', 'test'].")

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return loader
