import matplotlib.pyplot as plt

# Data for the pie chart (sorted in descending order)
disease_data = {
    'Unknown': 30, 
    'Blight': 17.53, 
    'Mosaic': 15.58, 
    'Mildew': 14.33, 
    'Rust': 12.07, 
    'Healthy': 10.49
}

# Sort data by percentage in descending order
disease_data = dict(sorted(disease_data.items(), key=lambda x: x[1], reverse=True))

labels = list(disease_data.keys())
sizes = list(disease_data.values())
colors = ['lightgray', 'lightgreen', 'lightblue', 'lightcoral', 'plum', 'lightyellow']

# Explode the largest section (Unknown) for emphasis
explode = [0.1 if label == "Unknown" else 0 for label in labels]

# Create the pie chart
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    sizes, labels=labels, autopct='%1.2f%%', colors=colors, startangle=140,
    wedgeprops={'edgecolor': 'black'}, explode=explode
)

# Improve text visibility
for text in autotexts:
    text.set_fontsize(10)
    text.set_weight('bold')

# Add a legend for better readability
plt.legend(wedges, labels, title="Diseases", loc="upper right", bbox_to_anchor=(1.3, 1))

# Title
plt.title('Plant Disease Distribution', fontsize=14, fontweight='bold')

# Save the chart
plt.savefig('plant_disease_distribution.png', dpi=300, bbox_inches='tight')

# Show the chart
plt.show()
