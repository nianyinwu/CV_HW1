import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset


class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.filename = [os.path.basename(f) for f in os.listdir(self.root)]


    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.filename[idx])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, self.filename[idx]


def dataloader(args, mode):

    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomChoice([
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3)
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    if mode == 'train':
        path = os.path.join(args.data_path, 'train')
        dataset = ImageFolder(root=path, transform=train_transform)
        shuffle = True
    elif mode == 'val':
        path = os.path.join(args.data_path, 'val')
        dataset = ImageFolder(root=path, transform=transform)
        shuffle = False
    elif mode == 'test':
        path = os.path.join(args.data_path, 'test')
        dataset = TestDataset(root=path, transform=transform)
        shuffle = False


    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    return loader
