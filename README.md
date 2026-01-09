# 🐾 WildLife — AI-Powered Wildlife Image Classifier

WildLife is a **Django-based web application** integrated with a **deep learning image classification pipeline**.  
It allows users to **register, log in, upload wildlife images**, and get predictions using a **trained ResNet model**.

This project combines:
- 🌐 Web development (Django)
- 🤖 Machine Learning (CNN / ResNet)
- 🎨 Frontend styling (HTML + CSS)
- 🧠 Model inference pipeline

---

## 🚀 Features

- 🔐 User Authentication (Register / Login)
- 📸 Image Upload for Classification
- 🧠 Deep Learning Model Integration (ResNet)
- 🗂 Modular Django App Structure
- 🎨 Separate static files per app
- ⚙️ Clean ML pipeline (train / evaluate / predict)

---

## 🧠 Tech Stack

**Backend**
- Python 3.12
- Django
- SQLite (development)

**Machine Learning**
- TensorFlow / Keras
- ResNet-based CNN

**Frontend**
- HTML5
- CSS3

---

## 📁 Project Structure

```text
E:.
├── manage.py
├── db.sqlite3
├── accounts
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── urls.py
│   ├── views.py
│   ├── migrations
│   └── templates
│       ├── base.html
│       └── accounts
│           ├── login.html
│           └── register.html
├── app
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── classifier
│   ├── views.py
│   ├── urls.py
│   ├── models.py
│   ├── migrations
│   └── templates
│       └── classifier
│           └── home.html
├── classifier_uploads
├── core
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── datasets
│   └── models
│       ├── best_resnet_model.keras
│       └── best_resnet_model.h5
└── static
    ├── accounts
    │   └── styles.css
    └── classifier
        └── styles.css
