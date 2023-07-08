from django.db import models

class UploadedImage(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    image = models.ImageField(upload_to='uploaded_images/')


class FaceEmbedding(models.Model):
    person_id = models.CharField(max_length=255)  
    person_name = models.CharField(max_length=255)
    embedding = models.BinaryField()
