import os

from rest_framework.views import APIView
from rest_framework.response import Response
from .face_recognition import train_face_recognition,test_single_image
from django.http import JsonResponse
from PIL import Image
import pickle
import json


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
        

        # Pass the directory path to the ML model
        train_face_recognition('./train_dataset')


        return Response({'message': 'Images uploaded successfully.'})
                                                                 

class UnknownImageUploadAPI(APIView):

    def post(self, request):
        image_data = request.FILES.get('image_data', None)
        if image_data is not None:
            # Convert uploaded image data to PIL Image
            image = Image.open(image_data)
            # Save the image temporarily in the face_test directory
            directory_path = os.path.join(os.path.dirname(__file__), 'face_test')
            os.makedirs(directory_path, exist_ok=True)
            image_path = os.path.join(directory_path, 'uploaded_image.jpg')
            image.save(image_path)
            # Pass the image path to the model for prediction

            file_path_1 = './face_recoginition/emb/embeddings.pickle'
            file_path_2 = './face_recoginition/emb/svm.pickle'
            file_path_3 = './face_recoginition/emb/label.pickle'

            # Load the pickled file
            with open(file_path_1, 'rb') as file:
                embeddings = pickle.load(file)

            with open(file_path_2, 'rb') as file:
                clf = pickle.load(file)

            with open(file_path_3, 'rb') as file:
                le = pickle.load(file)

            print(embeddings)
            print(clf)
            print(le)

            prediction = test_single_image(clf,le,embeddings)

            # Delete the temporary image file
            os.remove(image_path)

            # Return the prediction as JSON response
            return JsonResponse({'prediction': prediction})

        return JsonResponse({'error': 'Image data is missing.'})

class RetrieveModelStatus(APIView):
    def get(self, request):
            file_path = './face_recoginition/emb/embeddings.pickle'

            # Load the pickled file
            with open(file_path, 'rb') as file:
                embeddings = pickle.load(file)

            # embeddings = embeddings.tolist()

            # Print the retrieved content
            print(embeddings)

            # Convert the embeddings to JSON
            # embeddings_json = json.dumps(embeddings)

            # Return the JSON representation
            return embeddings

