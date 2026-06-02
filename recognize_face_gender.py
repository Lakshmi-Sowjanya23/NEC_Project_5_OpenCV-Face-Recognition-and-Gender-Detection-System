import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load gender model
gender_model = load_model("models/gender_model.h5")

# Load face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/face_recognizer.yml")

# Load name mapping
with open("models/name_mapping.pkl", "rb") as f:
    name_to_id = pickle.load(f)

id_to_name = {v: k for k, v in name_to_id.items()}

# Gender mapping
gender_map = {
    "babu": "Male",
    "souji": "Female",
    "revathi": "Female",
    "radhika": "Female"
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face_gray = gray[y:y+h, x:x+w]
        face_gray = cv2.resize(face_gray, (200, 200))

        # Name prediction
        person_id, distance = recognizer.predict(face_gray)

        person_name = id_to_name.get(person_id, "Unknown")

        # Gender prediction
        face_color = frame[y:y+h, x:x+w]
        face_color = cv2.resize(face_color, (64, 64))
        face_color = face_color / 255.0
        face_color = np.expand_dims(face_color, axis=0)

        gender_pred = gender_model.predict(face_color, verbose=0)

        gender_index = np.argmax(gender_pred)

        gender = ["Male", "Female"][gender_index]

        gender_confidence = np.max(gender_pred) * 100

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Name: {person_name}",
            (x, y-40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Gender: {gender}",
            (x, y-15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {gender_confidence:.1f}%",
            (x, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    cv2.imshow("Face Recognition + Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()