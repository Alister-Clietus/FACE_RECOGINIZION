import os
import cv2
import numpy as np
import insightface
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from .models import FaceEmbedding,ModelStatus
import pickle
import random
import string


def get_person_id_from_name(person_name):
    # Generate a random person ID based on their name
    random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return random_id


def train_face_recognition(train_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.8, 0.9, 0.9], factor=0.709, post_process=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    model = insightface.app.FaceAnalysis()
    ctx_id = 0  # to use GPU
    model.prepare(ctx_id=ctx_id)

    face_db = {}

    X_train = []
    y_train = []

    for person in os.listdir(train_dir):

        person_dir = os.path.join(train_dir, person)
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                face = mtcnn(image)

                if face is not None:
                    emb = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
                    X_train.append(emb)
                    y_train.append(person)

            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")

    X_train = np.concatenate(X_train)
    X_train = X_train.reshape(len(X_train), -1)
    y_train = np.array(y_train)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    clf = SVC(probability=True)
    clf.fit(X_train, y_train_enc)

    for person in os.listdir(train_dir):
        person_dir = os.path.join(train_dir, person)
        emb_list = []

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)

            faces = model.get(img)

            if faces:
                face = faces[0]
                emb = face.normed_embedding
                emb_list.append(emb)

        if emb_list:
            face_db[person] = np.mean(emb_list, axis=0)


    # Save the face embeddings in the database
    for person, embedding in face_db.items():
        person_id = get_person_id_from_name(person)  # Replace this with your logic to get the person's ID from the name
        embedding_instance = FaceEmbedding(person_id=person_id, person_name=person, embedding=embedding.tobytes())
        embedding_instance.save()

    return clf, le, face_db 


def test_single_image(clf, le, face_db):
    image_path='./face_app/face_test/uploaded_image.jpg'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.8, 0.9, 0.9], factor=0.709, post_process=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    THRESHOLD_INSIGHTFACE = 0.9  # Threshold for InsightFace
    THRESHOLD_FACENET = 1.0  # Threshold for Facenet

    model = insightface.app.FaceAnalysis()
    ctx_id = 0  # to use GPU
    model.prepare(ctx_id=ctx_id)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces_insight = model.get(image)
    if faces_insight:
        recognized_faces = []

        for face_insight in faces_insight:
            bbox = face_insight.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            face = extract_face(image, box=bbox, image_size=160)
            emb = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()
        
            prediction = clf.predict(emb)
            proba = clf.predict_proba(emb)

            person_name_facenet = le.inverse_transform(prediction)[0] if np.max(proba) > THRESHOLD_FACENET else "unknown"
            emb = face_insight.normed_embedding

            min_dist = 1000000  # Large arbitrary value
            person_name_insightface = 'unknown'

            for name, saved_emb in face_db.items():
                dist = np.linalg.norm(saved_emb - emb)

                if dist < min_dist:
                    min_dist = dist
                    person_name_insightface = name

            person_name_insightface = person_name_insightface if min_dist < THRESHOLD_INSIGHTFACE else 'unknown'

            if person_name_insightface == 'unknown' and person_name_facenet == 'unknown':
                person_name = 'unknown'
            elif person_name_insightface != 'unknown':
                person_name = person_name_insightface
            else:
                person_name = person_name_facenet

            labeled_image = image.copy()  # Copy image for each detected face

            color = (0, 255, 0) if person_name != "unknown" else (255, 0, 0)  # green for recognized, red for unknown
            text = "{}: {:.2f}%".format(person_name, np.max(proba) * 100)

            cv2.rectangle(labeled_image, (x1, y1), (x2, y2), color, 2)

            cv2.rectangle(labeled_image, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(labeled_image, text, (x1 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

            print('The face in "{}" recognized as {}'.format(image_path, person_name))

            if person_name != 'unknown':
                recognized_faces.append(person_name)

        print("Recognized Faces:", recognized_faces)

        return recognized_faces
    else:
        print("No faces detected in the image.")
        return []



# train_dir = './faces'
# image_path = './faces_test/mamooty.jpg'

# # Train face recognition models
# clf, le, face_db = train_face_recognition(train_dir)

# # Test single image
# recognized_faces = test_single_image(image_path, clf, le, face_db)