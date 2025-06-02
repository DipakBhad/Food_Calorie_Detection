import json

# List of food categories
food_categories = [
    'Bhindi_masala', 'Biryani', 'Dal_tadka', 'Jalebi', 'Kachori', 
    'Kofta', 'Pizza', 'Poha', 'Rasgulla', 'Roti'
]

# Create a dictionary mapping each category to an index
class_indices = {category: index for index, category in enumerate(food_categories)}

# Save the dictionary to a JSON file
json_filename = "class_indices(2).json"
with open(json_filename, 'w') as json_file:
    json.dump(class_indices, json_file, indent=4)

print(f"JSON file '{json_filename}' created successfully!")
