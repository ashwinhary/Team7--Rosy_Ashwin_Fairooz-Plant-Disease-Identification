# Team7--Rosy_Ashwin_Fairooz-Plant-Disease-Identification
# Plant Disease Classification using MobileNetV2

This project aims to classify plant species and detect diseases using a hierarchical deep learning model based on **MobileNetV2**. The model is trained to first classify plant species and then predict specific diseases affecting the identified species.

## 📌 Features
- **Hierarchical Classification**: First detects plant species, then predicts disease.
- **MobileNetV2 Backbone**: Lightweight and efficient pre-trained model.
- **Deep Supervision**: Separate loss functions for plant and disease classification.
- **Adaptive Loss Balancing**: Ensures effective multi-task learning.
- **Data Augmentation**: Improves generalization.
- **Visualization & Performance Metrics**: Includes confusion matrix, accuracy plots, and misclassification analysis.


## 🏗 Project Structure
📂 Plant-Disease-Classification 
 │── 📜 build_model.py # Defines the hierarchical MobileNetV2 model 
│── 📜 compile_model.py # Compiles the model with loss and optimizer 
│── 📜 train_mobilenet_plant_disease.py # Trains the model 
│── 📜 plant_disease_prediction.py # Runs inference on new images 
│── 📜 plant_disease_model_evaluation.py # Evaluates model performance 
│── 📜 plant_and_disease_classifier_with_visualization.py # Visualizes predictions 
│── 📜 plant_leaf_augmentation.py # Applies data augmentation 
│── 📜 train_val_test_split.py # Splits data into train/val/test sets 
│── 📜 load_preprocess.py # Preprocesses dataset 
│── 📜 confusion_matrix_and_classification_report_evaluation.py # Generates confusion matrix 
│── 📜 README.md # Project documentation


## 🖥 Installation & Setup

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/plant-disease-classification.git
   cd plant-disease-classification


## 🖥Install dependencies:
pip install -r requirements.txt

📂 dataset
├── 📂 train
│   ├── 📂 plant_1
│   │   ├── healthy.jpg
│   │   ├── diseased.jpg
│   ├── 📂 plant_2
│   │   ├── healthy.jpg
│   │   ├── diseased.jpg
├── 📂 test
├── 📂 validation


## Training the Model:
python train_mobilenet_plant_disease.py
Check models/ for saved weights.
 
 
##  Running Inference:
python plant_disease_prediction.py --image path/to/image.jpg

## Model Evaluation:
python plant_disease_model_evaluation.py
python confusion_matrix_and_classification_report_evaluation.py


## Results & Performance
Achieved high accuracy on plant and disease classification.
Improved generalization with data augmentation.
Efficient inference using MobileNetV2.


##  Future Improvements
Fine-tuning MobileNetV2 for better accuracy.
Adding more plant species and disease categories.
Deploying as a web or mobile app.

## License
MIT License © 2025 Team7

## Acknowledgments
Special thanks to the dataset providers and open-source community!









