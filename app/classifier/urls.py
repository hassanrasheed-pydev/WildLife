from django.urls import path
from . import views

app_name = 'classifier'  # This is your app namespace

urlpatterns = [
    path('', views.home, name='home'),  # or whatever your home view is
]