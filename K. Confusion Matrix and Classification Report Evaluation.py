import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load actual labels and model predictions
# Make sure you have saved the ground truth and predicted labels during your evaluation
# If not, make sure to modify your model to save them

try:
    y_true = np.load("/home/exouser/y_true.npy")  # Actual labels (from your test set)
    y_pred = np.load("/home/exouser/y_pred.npy")  # Predictions from your model

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=plant_classes, yticklabels=plant_classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Generate Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=plant_classes))

except FileNotFoundError:
    print("Confusion Matrix and Classification Report skipped: Test data not found.")
