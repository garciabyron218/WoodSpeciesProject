import matplotlib.pyplot as plt
import numpy as np

# Helper function to unnormalize and show an image
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))  # Rearrange from (C, H, W) to (H, W, C)
    mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
    std = np.array([0.229, 0.224, 0.225])   # ImageNet std
    inp = std * inp + mean                  # Unnormalize
    inp = np.clip(inp, 0, 1)                # Clip to valid pixel range
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')  # No axis ticks
    plt.show()

# Get a batch of training images
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs[:4], nrow=4)  # Display 4 images

# Display images with class names
class_names = train_dataset.classes
imshow(out, title=[class_names[x] for x in classes[:4]])
