import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Paths
train_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\split_images\train'
val_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\split_images\val'

# Define transformations (match your augmentations a little)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # standard ImageNet normalization
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Get number of classes
num_classes = len(train_dataset.classes)
print(f"✅ Loaded {len(train_dataset)} training images across {num_classes} classes.")
print(f"✅ Loaded {len(val_dataset)} validation images across {num_classes} classes.")
