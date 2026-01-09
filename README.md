# WildLife Explorer - AI-Powered Wildlife Search Engine

A Django-based web application that uses deep learning (ResNet) to classify wildlife images. Features user authentication, image upload capabilities, and real-time predictions.

---

## 📁 Project Structure

```
E:.
├── db.sqlite3
├── manage.py
├── accounts/
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   ├── views.py
│   ├── migrations/
│   │   └── __init__.py
│   └── templates/
│       ├── base.html
│       └── accounts/
│           ├── login.html
│           └── register.html
├── app/
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── classifier/
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   ├── views.py
│   ├── migrations/
│   │   └── 0001_initial.py
│   └── templates/
│       └── classifier/
│           └── home.html
├── classifier_uploads/
├── core/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── datasets/
│   └── models/
│       ├── best_resnet_model.keras
│       └── best_resnet_model.h5
└── static/
    ├── accounts/
    │   └── styles.css
    └── classifier/
        └── styles.css
```

---

## 🏗️ Architecture Overview

### 🔹 `manage.py`
Main entry point of the Django project used to run the development server, apply migrations, create superusers, and execute Django commands.

```bash
python manage.py runserver
```

### 🔹 `db.sqlite3`
SQLite database used during development. Stores user authentication data and Django model records.

### 🔐 `accounts/` — Authentication App
Handles all user authentication logic including user registration, login/logout functionality, and access control using `login_required` decorator.

### ⚙️ `app/` — Django Project Configuration
Core Django project settings:
- `settings.py` → Installed apps, middleware, static/media configuration
- `urls.py` → Root URL routing
- `wsgi.py` → Production server entry point
- `asgi.py` → Async support

### 🧠 `classifier/` — Image Classification App
Handles image uploads, calls the ML prediction pipeline, and renders prediction results through the `home.html` classifier UI.

### 📸 `classifier_uploads/`
Directory where uploaded images are stored. Acts as Django's MEDIA directory and is created automatically.

> 💡 **Tip:** Should not be committed to GitHub.

### 🤖 `core/` — Machine Learning Pipeline
Contains pure ML code independent of Django:
- `train.py` → Train CNN/ResNet model
- `evaluate.py` → Evaluate model performance
- `predict.py` → Load model and predict images
- `datasets/` → Training data
- `models/` → Saved trained models

This separation keeps the architecture clean and scalable.

### 🎨 `static/` — Static Files
Contains CSS files for styling different apps, loaded using Django's static file system.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/hassanrasheed-pydev/WildLife.git
cd WildLife
```

### 2️⃣ Create & activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Apply migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5️⃣ Run development server
```bash
python manage.py runserver
```

Open your browser at: **http://127.0.0.1:8000/**

---

## 🔒 Access Control

- Classifier page requires authentication
- Unauthorized users are redirected to login page
- Implemented using Django decorators

---

## 🚀 Future Improvements

- [ ] Prediction confidence scores
- [ ] Support for multiple wildlife species
- [ ] Prediction history per user
- [ ] Cloud storage for uploads (AWS S3, Google Cloud)
- [ ] Dockerized deployment
- [ ] API endpoints for mobile integration
- [ ] Model performance metrics dashboard

---

## 👨‍💻 Author

**Hassan Rasheed**  
ML Engineer | Django Developer | AI Enthusiast

---

## ⭐ Support

If you like this project, consider giving it a star ⭐  
It really helps and motivates further development!

---
