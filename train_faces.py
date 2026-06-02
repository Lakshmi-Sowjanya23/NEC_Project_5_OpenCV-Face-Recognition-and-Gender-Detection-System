import cv2
import os
import numpy as np
import pickle

dataset_path = "dataset"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

name_to_id = {}
current_id = 0

for gender_folder in ["male", "female"]:

    folder_path = os.path.join(dataset_path, gender_folder)

    for file in os.listdir(folder_path):

        if not file.endswith(".jpg"):
            continue

        person_name = file.split("_")[0]

        if person_name not in name_to_id:
            name_to_id[person_name] = current_id
            current_id += 1

        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in faces_detected:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            faces.append(face)
            labels.append(name_to_id[person_name])

recognizer.train(faces, np.array(labels))

recognizer.save("models/face_recognizer.yml")

with open("models/name_mapping.pkl", "wb") as f:
    pickle.dump(name_to_id, f)

print("Face recognizer trained successfully!")
print(name_to_id)