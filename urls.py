"""Recommendationsystem URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import index
##from . import UserDashboard
##from . import index
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('index', index.index, name='index'),
    path('register', index.register, name='register'),
    path('login', index.login, name='login'),
    path('user_dashboard', index.user_dashboard, name='user_dashboard'),
    path('logout', index.logout, name='logout'),
    path('update_profile', index.update_profile, name='update_profile'),
    path('predict_food',index.predict_food, name='predict_food'),
    path('body_calorie_calculator',index.body_calorie_calculator, name='body_calorie_calculator'),
    path('predict_calories',index.predict_calories, name='predict_calories'),
    path('food_insights_result',index.food_insights_result, name='food_insights_result'),
    path('track_calories', index.track_calories, name='track_calories'),
    path('calories_result', index.calories_result, name='calories_result'),

    
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)\
+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

