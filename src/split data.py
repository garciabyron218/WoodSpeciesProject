import os
import random
import shutil

# Input and Output Paths
input_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\augmented_images'
output_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\split_images'

train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')

# Create train and val folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split ratio
train_ratio = 0.7

# Loop through each species
for species_name in os.listdir(input_dir):
    species_folder = os.path.join(input_dir, species_name)
    if not os.path.isdir(species_folder):
        continue

    # List all images for that species
    images = [img for img in os.listdir(species_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) == 0:
        continue

    random.shuffle(images)

    split_point = int(len(images) * train_ratio)
    train_images = images[:split_point]
    val_images = images[split_point:]

    # Create species subfolders in train/val
    train_species_folder = os.path.join(train_dir, species_name)
    val_species_folder = os.path.join(val_dir, species_name)
    os.makedirs(train_species_folder, exist_ok=True)
    os.makedirs(val_species_folder, exist_ok=True)

    # Copy images to train
    for img_name in train_images:
        src = os.path.join(species_folder, img_name)
        dst = os.path.join(train_species_folder, img_name)
        shutil.copy2(src, dst)

    # Copy images to val
    for img_name in val_images:
        src = os.path.join(species_folder, img_name)
        dst = os.path.join(val_species_folder, img_name)
        shutil.copy2(src, dst)

print('✅ Dataset successfully split into training and validation sets!')
