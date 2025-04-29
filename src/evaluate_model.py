import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Settings
num_classes = 47
model_path = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\model_best.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

print("✅ Model loaded successfully!")

# Define transforms for the test image
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Classes
classes = os.listdir(r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\Data\split_images\train')
classes.sort()

# Function to predict a single image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transforms(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    outputs = model(image)
    _, preds = torch.max(outputs, 1)

    predicted_class = classes[preds[0]]
    return predicted_class

# Example usage:
test_image = r'C:\Users\garci\Documents\WoodSpecies\WoodSpecies\test\live-edge-cherry-slab-LE2217-1B-5.jpg'  # <-- Replace with a real path
prediction = predict_image(test_image)
print(f"🔎 Predicted species: {prediction}")
