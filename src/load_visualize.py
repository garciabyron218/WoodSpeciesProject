import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define paths
train_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\split_images\train'
val_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\split_images\val'

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Helper function to show images
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # Unnormalize
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Main logic
if __name__ == '__main__':
    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # ✅ Print loaded dataset info
    print(f"✅ Loaded {len(train_dataset)} training images across {len(train_dataset.classes)} classes.")
    print(f"✅ Loaded {len(val_dataset)} validation images across {len(val_dataset.classes)} classes.")

    # Get a batch of training data
    inputs, classes = next(iter(train_loader))

    # Make a grid of 4 images
    out = torchvision.utils.make_grid(inputs[:4], nrow=4)

    # Show images
    class_names = train_dataset.classes
    imshow(out, title=[class_names[x] for x in classes[:4]])
