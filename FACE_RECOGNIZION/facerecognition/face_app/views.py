import os
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import ImageUploadSerializer
from .face_recognition import train_face_recognition,test_single_image
from django.views import View
from django.http import JsonResponse
from PIL import Image
import io
import base64
import pickle
from .models import ModelStatus



clf = None
le = None
face_db = None

class KnownImageUploadAPI(APIView):
    def post(self, request, *args, **kwargs):
        name = request.data.get('name')  # Assuming 'name' is sent as part of the request data
        images = request.FILES.getlist('images')  # Assuming 'images' is the name of the file input field

        # Create a directory for the images
        directory_path = os.path.join('train_dataset', name)
        os.makedirs(directory_path, exist_ok=True)

        for image in images:
            # Save each image in the directory
            image_path = os.path.join(directory_path, image.name)
            with open(image_path, 'wb') as file:
                for chunk in image.chunks():
                    file.write(chunk)
        
        classifier_blob = pickle.dumps(clf)
        label_encoder_blob = pickle.dumps(le)

        # Pass the directory path to the ML model
        clf,le,face_db=train_face_recognition('./train_dataset')

        model_status, _ = ModelStatus.objects.get_or_create()
        model_status.classifier = classifier_blob
        model_status.label_encoder = label_encoder_blob
        model_status.face_database = face_db
        model_status.save()

        return Response({'message': 'Images uploaded successfully.'})
                                                                 

class UnknownImageUploadAPI(APIView):
    model_status = ModelStatus.objects.first()

    # Check if ModelStatus instance exists
    if model_status:
        clf = pickle.loads(model_status.classifier)
        le = pickle.loads(model_status.label_encoder)
        face_db = model_status.face_databas
    # clf = None
    # le = None
    # face_db = None
    # def get_model_status(self):
    #     if self.clf is None or self.le is None or self.face_db is None:
    #         try:
    #             model_status = ModelStatus.objects.latest('id')
    #             self.clf = pickle.loads(model_status.classifier)
    #             self.le = pickle.loads(model_status.label_encoder)
    #             self.face_db = model_status.face_database
    #             print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    #             print(self.clf)
    #             print(self.le)
    #             print(self.face_db)
    #         except ModelStatus.DoesNotExist:
    #             # Raise a custom exception
    #             raise ModelStatusNotFoundError("Model status not found")


    def post(self, request):
        image_data = request.FILES.get('image_data', None)
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print(clf)
        print(le)
        print(face_db)
        if image_data is not None:
            # Convert uploaded image data to PIL Image
            image = Image.open(image_data)
            # Save the image temporarily in the face_test directory
            directory_path = os.path.join(os.path.dirname(__file__), 'face_test')
            os.makedirs(directory_path, exist_ok=True)
            image_path = os.path.join(directory_path, 'uploaded_image.jpg')
            image.save(image_path)
            # Pass the image path to the model for prediction

            prediction = test_single_image(clf,le,face_db)

            # Delete the temporary image file
            os.remove(image_path)

            # Return the prediction as JSON response
            return JsonResponse({'prediction': prediction})

        return JsonResponse({'error': 'Image data is missing.'})

class RetrieveModelStatus(APIView):
    def get(self, request):
        try:
            model_status = ModelStatus.objects.latest('id')
            classifier = pickle.loads(model_status.classifier)
            label_encoder = pickle.loads(model_status.label_encoder)
            face_database = model_status.face_database

            # Print the classifier
            print(classifier)

            # Print the label encoder
            print(label_encoder)

            # Print the face database
            print(face_database)

            return Response({'message': 'Model status retrieved successfully.'})

        except ModelStatus.DoesNotExist:
            return Response({'error': 'Model status not found.'}, status=status.HTTP_404_NOT_FOUND)
    
