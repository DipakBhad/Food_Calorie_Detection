import os
import csv

# Define your dataset directory
dataset_dir = 'dataset/train_ten'

# List of food categories
food_categories = [
    'Bhindi_masala', 'Biryani', 'Dal_tadka', 'Jalebi', 'Kachori', 
    'Kofta', 'Pizza', 'Poha', 'Rasgulla', 'Roti'
]

# Nutritional values for each food category
nutrition_data = {
    "Bhindi_masala": {"calories": 120, "proteins": 3, "carbs": 10, "fats": 7},
    "Biryani": {"calories": 320, "proteins": 15, "carbs": 45, "fats": 10},
    "Dal_tadka": {"calories": 180, "proteins": 9, "carbs": 25, "fats": 5},
    "Jalebi": {"calories": 300, "proteins": 2, "carbs": 50, "fats": 12},
    "Kachori": {"calories": 170, "proteins": 4, "carbs": 20, "fats": 8},
    "Kofta": {"calories": 250, "proteins": 10, "carbs": 20, "fats": 15},
    "Pizza": {"calories": 285, "proteins": 12, "carbs": 36, "fats": 10},
    "Poha": {"calories": 180, "proteins": 4, "carbs": 30, "fats": 5},
    "Rasgulla": {"calories": 180, "proteins": 4, "carbs": 40, "fats": 2},
    "Roti": {"calories": 75, "proteins": 3, "carbs": 15, "fats": 1},
}

# Create CSV file
csv_filename = 'food_data_ten.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['filename', 'category', 'calories', 'proteins', 'carbs', 'fats'])

    # Loop through each food category and its images
    for category in food_categories:
        category_dir = os.path.join(dataset_dir, category)

        # Check if the directory exists
        if os.path.exists(category_dir):
            for image_name in os.listdir(category_dir):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
                    file_path = os.path.join(category, image_name)  # Relative path
                    nutrition = nutrition_data.get(category, {})
                    
                    # Get nutrition values, default to 0 if missing
                    calories = nutrition.get('calories', 0)
                    proteins = nutrition.get('proteins', 0)
                    carbs = nutrition.get('carbs', 0)
                    fats = nutrition.get('fats', 0)
                    
                    # Write row to CSV
                    writer.writerow([file_path, category, calories, proteins, carbs, fats])

print(f"CSV file '{csv_filename}' created successfully!")
