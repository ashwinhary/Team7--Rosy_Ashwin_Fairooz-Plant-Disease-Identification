import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Load the specific image from the provided path
image_path = "/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/test/Apple___Apple_scab/image (62).JPG"
img = load_img(image_path)  # Load image
img_array = img_to_array(img)  # Convert to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Create an ImageDataGenerator with the specified augmentations
datagen = ImageDataGenerator(
    rotation_range=40,         # Random rotations between 0 and 40 degrees
    width_shift_range=0.2,     # Random horizontal shifts (20% of the image width)
    height_shift_range=0.2,    # Random vertical shifts (20% of the image height)
    shear_range=0.2,           # Shear transformations
    zoom_range=0.2,            # Random zoom
    horizontal_flip=True,      # Random horizontal flip
    vertical_flip=True         # Random vertical flip (added explicitly)
)

# Generate augmented images (we'll generate 6 for display)
augmented_images = datagen.flow(img_array, batch_size=1)

# List of augmentation types for labeling
augmentation_types = [
    "Rotation", "Width Shift", "Height Shift", 
    "Shear", "Zoom", "Horizontal Flip", 
    "Vertical Flip"  # Explicitly include vertical flip
]

# Save the original and augmented images as a single image with labels
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# Original image
axes[0, 0].imshow(img)
axes[0, 0].set_title("Potato Leaf - Original Image")
axes[0, 0].axis('off')

# Show augmented images
for i in range(1, 7):  # Show 6 augmented images
    augmented_img = next(augmented_images)[0].astype('uint8')  # Use next() to get the augmented image
    ax = axes[i // 3, i % 3]  # Place images in a grid
    ax.imshow(augmented_img)
    ax.set_title(f"Potato Leaf - {augmentation_types[i-1]}")
    ax.axis('off')

# Remove any empty subplots (if any)
for j in range(7, 9):
    fig.delaxes(axes.flatten()[j])

# Save the plot as an image file
plt.tight_layout()
plt.savefig('/home/exouser/augmented_potato_leaf_images.png')  # Save the image
plt.close()
