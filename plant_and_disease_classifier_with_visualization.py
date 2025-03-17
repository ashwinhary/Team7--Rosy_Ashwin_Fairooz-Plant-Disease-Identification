import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Paths
dataset_path = "/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/"
model_plant_path = "/home/exouser/plant_type_model.h5"  # Model for plant type
model_disease_path = "/home/exouser/disease_model.h5"  # Model for diseases
json_path = "/home/exouser/Public/predictions.json"

# Load models
plant_model = load_model(model_plant_path)
disease_model = load_model(model_disease_path)

# Image size
img_size = (224, 224)
batch_size = 32

# Data generator for test images
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = test_datagen.flow_from_directory(
    os.path.join(dataset_path, "test"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

# Get class labels
plant_labels = list(test_gen.class_indices.keys())  # Plant types
disease_labels = ["Healthy", "Blight", "Rust", "Mosaic", "Mildew"]  # Update with actual disease labels

# Step 1: Predict Plant Type
plant_predictions = plant_model.predict(test_gen)
predicted_plant_classes = np.argmax(plant_predictions, axis=1)

# Step 2: Predict Disease Type
results = {}
for i, filename in enumerate(test_gen.filenames):
    plant_type = plant_labels[predicted_plant_classes[i]]

    # Load image for disease classification
    img = cv2.imread(os.path.join(dataset_path, "test", filename))
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Reshape for model

    # Predict Disease Type
    disease_prediction = disease_model.predict(img)
    predicted_disease_class = np.argmax(disease_prediction)
    disease_confidence = np.max(disease_prediction) * 100

    # Save Prediction
    results[filename] = {
        "Predicted Plant Type": plant_type,
        "Predicted Disease": disease_labels[predicted_disease_class],
        "Confidence": disease_confidence
    }

# Save results to JSON
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)

print("Predictions saved to JSON.")

# === PLOTTING RESULTS ===
# Count plant types & diseases
plant_type_counts = {}
disease_counts = {}

for _, data in results.items():
    plant_type_counts[data["Predicted Plant Type"]] = plant_type_counts.get(data["Predicted Plant Type"], 0) + 1
    disease_counts[data["Predicted Disease"]] = disease_counts.get(data["Predicted Disease"], 0) + 1

# Plot Plant Type Distribution
plt.figure(figsize=(12, 6))
plt.bar(plant_type_counts.keys(), plant_type_counts.values(), color="green")
plt.xlabel("Plant Type")
plt.ylabel("Count")
plt.title("Predicted Plant Type Distribution")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot Disease Distribution
plt.figure(figsize=(12, 6))
plt.bar(disease_counts.keys(), disease_counts.values(), color="red")
plt.xlabel("Disease Type")
plt.ylabel("Count")
plt.title("Predicted Disease Distribution")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("Analysis complete!")
