import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import custom_object_scope

# Define a dummy Cast layer (if required)
def Cast(*args, **kwargs):
    return tf.cast(*args, **kwargs)

# Load the model with custom object scope
with custom_object_scope({'Cast': Cast}):
    plant_model = load_model("plant_type_model.h5", compile=False)
    disease_model = load_model("plant_disease_model_final.h5", compile=False)

# ---------------------------
# Define paths
# ---------------------------
dataset_path = "/home/exouser/Public/Plant_leaf_diseases_dataset_with_augmentation/"
model_plant_path = "/home/exouser/plant_type_model.h5"  # Model for plant type
model_disease_path = "/home/exouser/plant_disease_model_final.h5"  # Model for diseases
json_path = "/home/exouser/Public/predictions.json"

# ---------------------------
# Load models with safeguards
# ---------------------------
try:
    plant_model = load_model(model_plant_path, compile=False)  # Load full model
    disease_model = load_model(model_disease_path, compile=False)  # Load full model
except ValueError as e:
    print("Error loading models:", e)
    exit(1)

# ---------------------------
# Prepare test generator
# ---------------------------
img_size = (224, 224)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = test_datagen.flow_from_directory(
    os.path.join(dataset_path, "test"),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

plant_labels = list(test_gen.class_indices.keys())
true_plant_labels = test_gen.labels  # True integer labels for plant types

# ---------------------------
# Predict plant type and evaluate performance
# ---------------------------
plant_predictions = plant_model.predict(test_gen)
predicted_plant_classes = np.argmax(plant_predictions, axis=1)

# Compute and visualize confusion matrix
cm = confusion_matrix(true_plant_labels, predicted_plant_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=plant_labels, yticklabels=plant_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for Plant Type")
plt.tight_layout()
plt.savefig("plant_confusion_matrix.png")
plt.close()

# Print classification report
report = classification_report(true_plant_labels, predicted_plant_classes, target_names=plant_labels)
print("Classification Report (Plant Type):\n", report)

# ---------------------------
# Predict disease type for each image
# ---------------------------
disease_labels = ["Healthy", "Blight", "Rust", "Mosaic", "Mildew"]  # Update if more classes exist

results = {}
for filename in test_gen.filenames:
    plant_type = filename.split('/')[0]

    img_path = os.path.join(dataset_path, "test", filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Unable to load image {img_path}")
        continue
    img = cv2.resize(img, img_size) / 255.0
    img = np.expand_dims(img, axis=0)

    disease_prediction = disease_model.predict(img)
    predicted_disease_class = np.argmax(disease_prediction)

    # Handle mismatched index errors
    disease_label = disease_labels[predicted_disease_class] if predicted_disease_class < len(disease_labels) else "Unknown"
    disease_confidence = float(np.max(disease_prediction) * 100)

    results[filename] = {
        "Predicted Plant Type": plant_type,
        "Predicted Disease": disease_label,
        "Confidence": disease_confidence
    }

# Save results to JSON
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)
print("\nDisease predictions saved to JSON.")

# ---------------------------
# Visualize Disease Distribution
# ---------------------------
with open(json_path, "r") as f:
    json_results = json.load(f)

disease_counts = {}
for _, data in json_results.items():
    disease_counts[data["Predicted Disease"]] = disease_counts.get(data["Predicted Disease"], 0) + 1

# Pie chart for disease predictions
plt.figure(figsize=(8, 8))
plt.pie(disease_counts.values(), labels=disease_counts.keys(), autopct='%1.1f%%', startangle=140)
plt.title("Disease Prediction Distribution")
plt.tight_layout()
plt.savefig("disease_distribution_pie.png")
plt.close()

print("Analysis complete! All results saved.")
