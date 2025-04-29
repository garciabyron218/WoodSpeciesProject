import os

# Paths
input_dir = r'/Data/augmented_images'
output_dir = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\split_images'

train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')

# Just make the folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print("✅ Created split_images/train and split_images/val folders!")
