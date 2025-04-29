import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ----------------------------
# SETTINGS
# ----------------------------

# Paths
train_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\split_images\train'
val_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\split_images\val'

# Parameters
num_classes = 47
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# ----------------------------
# DATA LOADING
# ----------------------------

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

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"✅ Loaded {len(train_dataset)} training images across {num_classes} classes.")
print(f"✅ Loaded {len(val_dataset)} validation images across {num_classes} classes.")

# ----------------------------
# MODEL SETUP
# ----------------------------

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # updated to use 'weights' properly
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

print("✅ Model is ready!")

# ----------------------------
# TRAINING FUNCTION
# ----------------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\n🔵 Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\n⏱️ Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Best Validation Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# ----------------------------
# TRAIN THE MODEL
# ----------------------------

trained_model = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs=num_epochs
)

# ----------------------------
# SAVE THE BEST MODEL
# ----------------------------

model_save_path = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\model_best.pth'
torch.save(trained_model.state_dict(), model_save_path)
print(f"✅ Best model saved to {model_save_path}")
