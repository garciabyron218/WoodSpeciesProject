import torch
import torch.nn as nn
import torchvision.models as models

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Number of classes (47 for your species)
num_classes = 47

# Load a pretrained ResNet18 model
model = models.resnet18(pretrained=True)

# Freeze early layers if you want (optional)
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to the device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

print("✅ Model is ready for training!")
