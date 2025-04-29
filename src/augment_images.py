import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.utils

# Paths
input_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\cropped_images'
output_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\augmented_images'
os.makedirs(output_dir, exist_ok=True)

# Define GENTLE transformations for wood images
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),     # 50% chance to flip horizontally
    transforms.RandomVerticalFlip(p=0.2),       # 20% chance to flip vertically
    transforms.RandomRotation(5),               # Very small rotation (5 degrees max)
    transforms.ColorJitter(brightness=0.1),      # Very small brightness change only
    transforms.ToTensor()
])

# Load dataset
dataset = ImageFolder(root=input_dir, transform=data_transforms)

# DataLoader to iterate over images
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# How many augmentations you want per image
num_augmented_images_per_image = 5

# Generate and save augmented images
for idx, (image, label) in enumerate(dataloader):
    species_name = dataset.classes[label.item()]
    species_dir = os.path.join(output_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)

    for i in range(num_augmented_images_per_image):
        augmented_image_path = os.path.join(species_dir, f"{species_name}_{idx}_{i}.jpg")
        torchvision.utils.save_image(image, augmented_image_path)
        print(f"Saved augmented image: {augmented_image_path}")
