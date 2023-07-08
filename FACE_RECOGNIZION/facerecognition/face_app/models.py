from django.db import models

# Create your models here.
from django.db import models
import pickle

class UploadedImage(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    image = models.ImageField(upload_to='uploaded_images/')


class FaceEmbedding(models.Model):
    person_id = models.CharField(max_length=255)  
    person_name = models.CharField(max_length=255)
    embedding = models.BinaryField()

class ModelStatus(models.Model):
    classifier = models.BinaryField(null=True, blank=True)
    label_encoder = models.BinaryField(null=True, blank=True)
    face_database = models.JSONField(null=True, blank=True)

    def __str__(self):
        return "Model Status"