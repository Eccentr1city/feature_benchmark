import json
import matplotlib.pyplot as plt

# Load the JSON data from the file
with open('6-res-jb_subset_100/428.json', 'r') as file:
    data = json.load(file)

# Initialize lists to store the IDs, recomputed values, and original values
data_by_id = []

# Loop through each entry in the posActivations
for activation in data['posActivations']:
    entry = {
        'id': activation['id'],
        'recomputed_values': activation['recomputedValues'][1:],  # Remove the first element from recomputed values
        'values': activation['values']
    }
    data_by_id.append(entry)

# Print each id with its corresponding recomputed values and values

for item in data_by_id:
    plt.figure(figsize=(10, 5))
    plt.plot(item['values'], label='Values', marker='o')
    plt.plot(item['recomputed_values'], label='Recomputed Values', marker='x')
    plt.title(f"Graph for ID: {item['id']}")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

