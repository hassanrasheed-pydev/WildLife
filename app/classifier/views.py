import os
from django.conf import settings
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.db.models import Q
import tensorflow as tf
from core.predict import predict_image
from .models import Animals

MODEL_PATH = os.path.join(
    settings.BASE_DIR,  
    "core",
    "models",
    "best_resnet_model (1).keras"
)

CLASS_NAMES = [
    "Horse", "bald_eagle", "black_bear", "bobcat", "cheetah", "cougar",
    "deer", "elk", "gray_fox", "hyena", "lion", "raccoon",
    "red_fox", "rhino", "tiger", "wolf", "zebra",
]

model = tf.keras.models.load_model(MODEL_PATH)

@login_required
def home(request):
    context = {}
    
    # Handle text search
    search_query = request.GET.get('search', '')
    if search_query:
        # Search for exact match first, then case-insensitive
        animal = Animals.objects.filter(name__iexact=search_query).first()
        
        if not animal:
            # Try partial match
            animal = Animals.objects.filter(name__icontains=search_query).first()
        
        context['search_query'] = search_query
        
        if animal:
            context['animal_info'] = animal
        else:
            context['animal_info'] = None
    
    # Handle image upload
    if request.method == "POST" and request.FILES.get("image"):
        img = request.FILES["image"]

        img_path = os.path.join(settings.MEDIA_ROOT, img.name)
        with open(img_path, "wb+") as f:
            for chunk in img.chunks():
                f.write(chunk)

        pred_class, probs = predict_image(
            model=model,
            image_path=img_path,
            class_names=CLASS_NAMES,
            plot=False
        )
        
        print(pred_class)
        print(probs)
        
        # Try exact match first
        animal_info = Animals.objects.filter(name__iexact=pred_class).first()
        
        if not animal_info:
            # Try case-insensitive partial match
            animal_info = Animals.objects.filter(name__icontains=pred_class).first()
        
        context['animal_info'] = animal_info
        context['prediction'] = pred_class
        context['probabilities'] = list(zip(CLASS_NAMES, probs))
        context['uploaded_image'] = img.name

    return render(request, "classifier/home.html", context)