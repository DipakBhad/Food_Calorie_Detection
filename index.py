from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.shortcuts import redirect, render
from django.template import RequestContext
import pymysql
from datetime import date
from django.core.mail import EmailMessage
from django.contrib import messages
from django.template.loader import render_to_string
from django.conf import settings
import random
import tensorflow as tf
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.preprocessing import image
from datetime import datetime
from tensorflow import keras
import json
import os

#database Connection
import pymysql
foodcaloriesdb=pymysql.connect(host="localhost",user="root",password="Pass@1234",database="food_calories_prediction_db")
cursor=foodcaloriesdb.cursor()

# Load the model and class indices when the server starts
# model = tf.keras.models.load_model('C:\\Users\\Asus\\OneDrive\\Desktop\\Nutritrack_Project\\code\\code\\Recommendationsystem\\food_classifier_nutrition_model_v2.h5')

model = keras.models.load_model("C:\\Users\\Asus\\OneDrive\\Desktop\\Nutritrack_Project\\code\\code\\Recommendationsystem\\food_classifier_nutrition_model_v2.h5", compile=False)

# Load class indices
with open('C:\\Users\\Asus\\OneDrive\\Desktop\\Nutritrack_Project\\code\\code\\Recommendationsystem\\class_indices_ten_v2.json', 'r') as f:
    class_indices = json.load(f)


def track_calories(request):
    if request.method == 'POST' and 'food_image_track' in request.FILES:
        food_image = request.FILES['food_image_track']

        # Save the uploaded image to a temporary location
        fs = FileSystemStorage()
        filename = fs.save(food_image.name, food_image)
        file_path = fs.url(filename)  # This provides the URL for the file

        # Preprocess the image for prediction
        img_path = os.path.join(os.getcwd(), 'media', filename)  # Full path to the image
        img = image.load_img(img_path, target_size=(224, 224))  # Resize image to (224x224)
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension for prediction

        # Predict the food category and calories
        prediction = model.predict(img_array)  # This should return a list or tuple
        
        # Adjust this part depending on how the model's predict method returns values
        if len(prediction) == 2:  # If it returns a tuple or list with 2 elements
            category_pred, calories_pred = prediction
        else:
            # Handle case if prediction doesn't return exactly 2 outputs
            category_pred = prediction[0]
            calories_pred = prediction[1]

        # Get the predicted food category and calories consumed
        predicted_category = class_names[np.argmax(category_pred)]
        calories_consumed = float(calories_pred[0][0])

        # Get the current datetime (when the food was uploaded)
        current_datetime = datetime.now()

        # Retrieve the user's calorie requirement from the session (or from the database if needed)
        calories_needed = float(request.session.get('calories_needed', 0))

        # Subtract the consumed calories from the user's calorie requirement
        updated_calories_needed = calories_needed - calories_consumed

        # Store the updated calorie value back to the session or database
        request.session['calories_needed'] = updated_calories_needed  # Store updated calories_needed in session

        # Calculate remaining calories the user can consume
        remaining_calories = updated_calories_needed

        # Store the tracked calories in the database (for persistence)
        try:
            cursor.execute("""
                INSERT INTO tracked_calories (food_name, calories_consumed, calories_needed, remaining_calories, date_time)
                VALUES (%s, %s, %s, %s, %s)
            """, (predicted_category, calories_consumed, updated_calories_needed, remaining_calories, current_datetime))

            # Commit the transaction to save the data
            foodcaloriesdb.commit()

        except Exception as e:
            foodcaloriesdb.rollback()
            print(f"Error occurred while saving data: {str(e)}")

        # Redirect to the result page and pass along the updated results
        return redirect(f'calories_result?calories_consumed={calories_consumed}&remaining_calories={remaining_calories}&food_name={predicted_category}&file_path={file_path}')

    return render(request, 'track_my_calories.html')


# Calories result view (show the results in a new page)
def calories_result(request):
    # Retrieve the passed parameters from the URL
    calories_consumed = request.GET.get('calories_consumed') #float(request.GET.get('calories_consumed'))
    remaining_calories = request.GET.get('remaining_calories') #float(request.GET.get('remaining_calories'))
    food_name = request.GET.get('food_name')
    file_path = request.GET.get('file_path')

    # Retrieve the user's calorie requirement from the session
    calories_needed = float(request.session.get('calories_needed', 0))

    # Check if the user consumed more calories than needed
    #excess_calories = calories_consumed - calories_needed
    excess_calories = - calories_needed

    # Pass all data to the template
    return render(request, 'calories_result.html', {
        'calories_consumed': calories_consumed,
        'remaining_calories': remaining_calories,
        'food_name': food_name,
        'file_path': file_path,
        'calories_needed': calories_needed,
        'excess_calories': excess_calories  # Flag to show if excess calories
    })

def predict_calories(request):
    if request.method == 'POST':
        # Extract form data
        name = request.POST.get('name')  # Capture the full name
        age = int(request.POST.get('age'))  # Age is still an integer
        gender = request.POST.get('gender')
        height = float(request.POST.get('height'))  
        weight = float(request.POST.get('weight'))  
        activity_level = request.POST.get('activity_level')

        # Calculate Basal Metabolic Rate (BMR) based on the Harris-Benedict Equation
        if gender == 'male':
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        elif gender == 'female':
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        else:
            # For 'other' gender, treat the BMR similarly to female or male
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

        # Adjust BMR based on activity level
        if activity_level == 'sedentary':
            tdee = bmr * 1.2  # Sedentary: little to no exercise
        elif activity_level == 'lightly active':
            tdee = bmr * 1.375  # Lightly Active: exercise 1-3 days/week
        elif activity_level == 'moderately active':
            tdee = bmr * 1.55  # Moderately Active: exercise 3-5 days/week
        elif activity_level == 'very active':
            tdee = bmr * 1.725  # Very Active: hard exercise 6-7 days/week
        elif activity_level == 'super active':
            tdee = bmr * 1.9  # Super Active: very intense exercise or physical job
        else:
            tdee = bmr * 1.2  # Default to sedentary if no valid input

        # Round the result to avoid too many decimal places
        calories_needed = round(tdee, 2)

        # Food Recommendation Logic
        if calories_needed < 1500:
            food_recommendation = """
<h3>Looking to reduce calorie intake? Here are some healthy food choices:</h3>
<ul>
    <li><strong>Lean Proteins:</strong> Skinless chicken, turkey, fish, egg whites, tofu</li>
    <li><strong>Fruits:</strong> Apples, berries, oranges, watermelon, pears</li>
    <li><strong>Vegetables:</strong> Cucumbers, leafy greens, carrots, bell peppers, zucchini</li>
    <li><strong>Whole Grains:</strong> Quinoa, brown rice (in moderation), whole wheat wraps</li>
    <li><strong>Healthy Snacks:</strong> Greek yogurt, cottage cheese, almonds (small portions)</li>
</ul>
<p>Incorporate these foods into your meals to maintain a healthy, low-calorie diet.</p>
"""
        elif 1500 <= calories_needed <= 2000:
            food_recommendation = """
<h3>Looking for a balanced diet? Here are some great food options:</h3>
<ul>
    <li><strong>Proteins:</strong> Chicken breast, fish, eggs, tofu, lentils, beans</li>
    <li><strong>Carbohydrates:</strong> Brown rice, quinoa, whole wheat bread, oats, sweet potatoes</li>
    <li><strong>Vegetables:</strong> Spinach, broccoli, carrots, bell peppers, tomatoes</li>
    <li><strong>Healthy Fats:</strong> Nuts, seeds, olive oil, avocado</li>
    <li><strong>Dairy:</strong> Low-fat yogurt, milk, cheese</li>
</ul>
<p>Include a mix of these in your meals to maintain a healthy and moderate calorie intake.</p>
"""
        else:
            food_recommendation = """
<h3>Want to increase your calorie intake? Try these nutritious foods:</h3>
<ul>
    <li><strong>Proteins:</strong> Chicken, turkey, lean beef, fish, eggs, tofu, lentils, beans</li>
    <li><strong>Carbohydrates:</strong> Brown rice, quinoa, oats, whole wheat bread, sweet potatoes, pasta</li>
    <li><strong>Healthy Fats:</strong> Avocados, olive oil, almonds, walnuts, chia seeds, flaxseeds</li>
    <li><strong>Dairy:</strong> Greek yogurt, cottage cheese, milk, cheese</li>
</ul>
"""

        # Get the current date and time
        current_datetime = datetime.now()

        # Store the calculated calories_needed in session
        request.session['calories_needed'] = calories_needed

        # Insert data into the database using raw SQL
        try:
            cursor.execute("""
                INSERT INTO user_calorie_predictions (full_name, age, gender, height, weight, activity_level, calories_needed, date_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (name, age, gender, height, weight, activity_level, calories_needed, current_datetime))

            # Commit the transaction to save the data
            foodcaloriesdb.commit()

        except Exception as e:
            foodcaloriesdb.rollback()
            print(f"Error occurred while saving data: {str(e)}")

        # Pass the calculated result along with the user's name, calories, and food recommendation to the result page
        return render(request, 'body_calorie_calculator.html', {
            'calories_needed': calories_needed,
            'user_name': name,  # Passing the user's name to the template
            'food_recommendation': food_recommendation,  # Passing the food recommendation to the template
        })

    return redirect('body_calorie_calculator')


def body_calorie_calculator(request):
    return render(request, "body_calorie_calculator")

#Food insights start
# Reverse the class indices to get class names
class_names = {v: k for k, v in class_indices.items()}

def predict_food(request):
    if request.method == 'POST' and request.FILES.get('food_image'):
        food_image = request.FILES['food_image']

        # Save the uploaded image
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(food_image.name, food_image)

        # Get the correct media URL path
        file_path = settings.MEDIA_URL + filename 

        # Preprocess the image for prediction
        img_path = os.path.join(settings.MEDIA_ROOT, filename)  # Get full path for image
        img = image.load_img(img_path, target_size=(224, 224))  # Resize image to 224x224
        img_array = image.img_to_array(img) / 255.0  # Normalize image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict food category, calories, proteins, carbs, and fats
        predictions = model.predict(img_array)

        category_pred = predictions[0]  # Softmax output for category
        calories_pred = predictions[1]
        proteins_pred = predictions[2]
        carbs_pred = predictions[3]
        fats_pred = predictions[4]

        # Get predicted category
        predicted_category = class_names[np.argmax(category_pred)]

        # Convert numpy values to Python floats
        predicted_calories = float(calories_pred[0][0])
        predicted_proteins = float(proteins_pred[0][0])
        predicted_carbs = float(carbs_pred[0][0])
        predicted_fats = float(fats_pred[0][0])

        # Pass data to the template
        return render(request, 'food_insights_result.html', {
            'food_name': predicted_category,
            'food_calories': predicted_calories,
            'food_proteins': predicted_proteins,
            'food_carbs': predicted_carbs,
            'food_fats': predicted_fats,
            'file_path': file_path
        })

    return render(request, 'index.html')

def food_insights_result(request):
    food_name = request.session.get('food_name')
    food_calories = request.session.get('food_calories')
    food_proteins = request.session.get('food_proteins')
    food_carbs = request.session.get('food_carbs')
    food_fats = request.session.get('food_fats')
    file_path = request.session.get('file_path')

    return render(request, 'food_insights_result.html', {
        'food_name': food_name,
        'food_calories': food_calories,
        'food_proteins': food_proteins,
        'food_carbs': food_carbs,
        'food_fats': food_fats,
        'file_path': file_path,
    })

def index(request):
    return render(request,"index.html")

# Registration View
def register(request):
    if request.method == 'POST':
        # Get the form data from POST request
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        contact = request.POST.get('contact')
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        height = request.POST.get('height')
        weight = request.POST.get('weight')
        activity_level = request.POST.get('activity_level')

        # Check if all fields are provided
        if not all([name, email, password, contact, age, gender, height, weight, activity_level]):
            messages.error(request, "All fields are required.")
            return redirect('register')

        # Check if the user already exists
        cursor.execute("SELECT * FROM user_registration_details WHERE email = %s", (email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            messages.error(request, "This email is already registered.")
            return redirect('register')

        # Insert the new user into the database
        try:
            cursor.execute("""
                INSERT INTO user_registration_details 
                (name, email, password, contact, age, gender, height, weight, activity_level) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (name, email, password, contact, age, gender, height, weight, activity_level))
            foodcaloriesdb.commit()
            messages.success(request, "Registration successful!")
            return redirect(reverse('index') + "#login")  # Redirect to index after registration
        except Exception as e:
            foodcaloriesdb.rollback()
            messages.error(request, f"Error occurred: {str(e)}")
            return redirect('register')

    return render(request, "index.html")


# Login View
def login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        if not email or not password:
            messages.error(request, "Email and password are required.")
            return redirect('index')

        cursor.execute("SELECT * FROM user_registration_details WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()

        if user:
            request.session['user_id'] = user[0]  # Store user ID in session
            request.session['user_email'] = user[2]
            return redirect('user_dashboard')
        else:
            messages.error(request, "Invalid credentials, please try again.")
            return redirect('index')

    return render(request, "index.html")

def user_dashboard(request):
    if 'user_id' not in request.session:
        messages.error(request, "You must be logged in to access the dashboard.")
        return redirect('index')

    user_id = request.session['user_id']
    cursor.execute("SELECT * FROM user_registration_details WHERE id = %s", (user_id,))
    user = cursor.fetchone()

    if user:
        user_data = {
            'name': user[1],
            'email': user[2],
            'contact': user[4],
            'age': user[5],
            'gender': user[6],
            'height': user[7],
            'weight': user[8],
            'activity_level': user[9]
        }
        return render(request, "user_dashboard.html", {'user': user_data})
    else:
        messages.error(request, "User not found.")
        return redirect('index')
    
# Update User Details
def update_profile(request):
    if request.method == 'POST':
        user_id = request.session.get('user_id')  # Get logged-in user ID from session
        name = request.POST.get('name')
        contact = request.POST.get('contact')
        age = request.POST.get('age')
        height = request.POST.get('height')
        weight = request.POST.get('weight')
        activity_level = request.POST.get('activity_level')
        gender = request.POST.get('gender')
        password = request.POST.get('password')  # Plain text password

        try:
            # Check if user exists
            cursor.execute("SELECT id FROM user_registration_details WHERE id = %s", (user_id,))
            user_exists = cursor.fetchone()

            if user_exists:
                if password:  # Update with new password if provided
                    cursor.execute("""
                        UPDATE user_registration_details 
                        SET name=%s, contact=%s, age=%s, height=%s, weight=%s, 
                            activity_level=%s, gender=%s, password=%s
                        WHERE id=%s
                    """, (name, contact, age, height, weight, activity_level, gender, password, user_id))
                else:  # Update without changing password
                    cursor.execute("""
                        UPDATE user_registration_details 
                        SET name=%s, contact=%s, age=%s, height=%s, weight=%s, 
                            activity_level=%s, gender=%s
                        WHERE id=%s
                    """, (name, contact, age, height, weight, activity_level, gender, user_id))

                foodcaloriesdb.commit()  # Save changes
                messages.success(request, "Profile updated successfully!")
            else:
                messages.error(request, "User not found!")

        except Exception as e:
            foodcaloriesdb.rollback()  # Rollback in case of error
            messages.error(request, f"Error: {str(e)}")

        return redirect(reverse('user_dashboard') + "#update_profile")

    # Fetch current user details for display
    user_id = request.session.get('user_id')
    cursor.execute("""
        SELECT name, email, contact, age, height, weight, activity_level, gender 
        FROM user_registration_details WHERE id = %s
    """, (user_id,))
    user_data = cursor.fetchone()

    # Convert tuple result into a dictionary for easy template rendering
    if user_data:
        user = {
            "name": user_data[0],
            "email": user_data[1],
            "contact": user_data[2],
            "age": user_data[3],
            "height": user_data[4],
            "weight": user_data[5],
            "activity_level": user_data[6],
            "gender": user_data[7],
        }
    else:
        user = None

    return render(request, "user_dashboard", {"user": user})

def logout(request):
    return render(request,"index.html")

