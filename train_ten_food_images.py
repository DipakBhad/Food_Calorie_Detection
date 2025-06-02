import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import Sequence, to_categorical
import json


# Custom data generator for multi-output tasks (food category + calories)
class MultiOutputDataGenerator(Sequence):
    def __init__(self, dataframe, directory, batch_size, target_size, x_col, y_col, shuffle=True):
        self.dataframe = dataframe
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.x_col = x_col
        self.y_col = y_col
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()

        # One-hot encoding for food categories
        self.category_indices = {category: index for index, category in enumerate(self.dataframe['category'].unique())}
        self.dataframe['category'] = self.dataframe['category'].map(self.category_indices)

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indexes]

        images = []
        categories = []
        calories = []
        proteins = []
        carbs = []
        fats = []
        for _, row in batch_data.iterrows():
            img_path = os.path.join(self.directory, row[self.x_col])
            img = image.load_img(img_path, target_size=self.target_size)
            img_array = image.img_to_array(img) / 255.0

            images.append(img_array)
            categories.append(row['category'])  # Already integer-mapped
            calories.append(row['calories'])
            proteins.append(row['proteins'])
            carbs.append(row['carbs'])
            fats.append(row['fats'])

        images = np.array(images)
        categories = np.array(categories)
        calories = np.array(calories)
        proteins = np.array(proteins)
        carbs = np.array(carbs)
        fats = np.array(fats)
        

        # One-hot encode categories
        categories = to_categorical(categories, num_classes=len(self.category_indices))

        return images, [categories, calories, proteins, carbs, fats]

# Load CSV data
dataframe = pd.read_csv('food_data_ten.csv')

# Set up directories for training and testing
train_dir = 'dataset/train_ten'
test_dir = 'dataset/test_ten'

# Create custom data generators for training and validation
train_generator = MultiOutputDataGenerator(
    dataframe=dataframe,
    directory=train_dir,
    batch_size=32,
    target_size=(224, 224),
    x_col='filename',
    y_col=['category', 'calories', 'proteins', 'carbs', 'fats'],
    shuffle=True
)

test_generator = MultiOutputDataGenerator(
    dataframe=dataframe,
    directory=test_dir,
    batch_size=32,
    target_size=(224, 224),
    x_col='filename',
    y_col=['category', 'calories','proteins', 'carbs', 'fats'],
    shuffle=False
)

# Build the CNN model
input_layer = layers.Input(shape=(224, 224, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)

# Output layers
# Output for food category (classification)
food_category_output = layers.Dense(len(dataframe['category'].unique()), activation='softmax', name='food_category')(x)
# regression
calorie_output = layers.Dense(1, name='calories')(x)
protein_output = layers.Dense(1, name='proteins')(x)
carb_output = layers.Dense(1, name='carbs')(x)
fat_output = layers.Dense(1, name='fats')(x)

# Create model
model = models.Model(inputs=input_layer, outputs=[food_category_output, calorie_output, protein_output, carb_output, fat_output])

# Compile model
model.compile(
    optimizer='adam',
    loss={'food_category': 'categorical_crossentropy', 'calories': 'mse', 'proteins': 'mse', 'carbs': 'mse', 'fats': 'mse'},
    metrics={'food_category': 'accuracy', 'calories': 'mae', 'proteins': 'mae', 'carbs': 'mae', 'fats': 'mae'}
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.__len__(),
    epochs=100,
    validation_data=test_generator,
    validation_steps=test_generator.__len__()
)

# Save the model
model.save('food_classifier_nutrition_model_v2.h5')

# Save the class indices to a JSON file
with open('class_indices_ten_v2.json', 'w') as f:
    json.dump(train_generator.category_indices, f)

print("Model training complete and saved as 'food_classifier_nutrition_model_v2.h5'")



