import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

# 1. Load Data from JSON
def load_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None  # Handle the error gracefully

def create_dataframe(data):
    """Creates a Pandas DataFrame from the loaded data."""
    if data is None:
        return None  # Exit if no data was loaded

    # Initialize lists to store plant types and confidences
    plant_types = []
    confidences = []

    for entry in data.values():  # Loop through each entry in the dictionary
        plant_type = entry.get('Predicted Plant Type')  # Get the plant type
        confidence = entry.get('Confidence')  # Get the confidence value
        if plant_type is not None and confidence is not None:
            plant_types.append(plant_type)
            confidences.append(confidence)

    df = pd.DataFrame({'Plant Type': plant_types, 'Confidence (%)': confidences})
    return df

# 2. Create the Box Plot
def create_box_plot(df, output_path="improved_box_plot.png"):
    """Creates and saves a box plot."""
    if df is None:
        return  # Exit if no DataFrame was created

    plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
    sns.boxplot(x='Plant Type', y='Confidence (%)', data=df, palette='viridis')  # Use 'viridis' or a similar palette

    plt.title('Confidence by Predicted Plant Type', fontsize=16)
    plt.xlabel('Plant Type', fontsize=12)
    plt.ylabel('Confidence (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels

    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(output_path, dpi=300)  # Save the plot
    plt.show()  # Display the plot (optional)

# 3. Main Execution
if __name__ == "__main__":
    json_file_path = '/home/exouser/Public/predictions.json'  # Updated JSON file path
    data = load_data(json_file_path)
    df = create_dataframe(data)
    if df is not None:
        create_box_plot(df)
        print("Box plot created and saved as 'improved_box_plot.png'")
    else:
        print("Box plot creation failed.")
