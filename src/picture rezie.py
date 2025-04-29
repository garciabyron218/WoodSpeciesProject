from PIL import Image
import os

def crop_images_in_subfolders(input_root_dir, output_root_dir):
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)

    for subdir, _, files in os.walk(input_root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(subdir, input_root_dir)
                output_subdir = os.path.join(output_root_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, file)
                with Image.open(input_path) as img:
                    width, height = img.size
                    left = width / 3
                    right = 2 * width / 3
                    top = height / 3
                    bottom = height
                    cropped_img = img.crop((left, top, right, bottom))
                    cropped_img.save(output_path)
                    print(f"Cropped and saved: {output_path}")

# Example usage:
input_root_directory = r'C:\Users\garci\Documents\WoodSpecies\.venv\Scripts\wood_species_images'
output_root_directory = r'C:\Users\garci\Documents\WoodSpecies\.venv\Scripts\cropped_images'
crop_images_in_subfolders(input_root_directory, output_root_directory)
