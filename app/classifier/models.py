from django.db import models

# Create your models here.
class Animals(models.Model):
    name = models.CharField(max_length=120)
    data = models.JSONField()
