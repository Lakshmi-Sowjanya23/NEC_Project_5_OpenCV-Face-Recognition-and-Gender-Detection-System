import cv2
import os

name = input("Enter Name: ")
gender = input("Enter Gender (male/female): ").lower()

folder = f"dataset/{gender}"

if not os.path.exists(folder):
    os.makedirs(folder)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
cap = cv2.VideoCapture(0)

count = 0
max_images = 50

print(f"Capturing {max_images} face images...")

while count < max_images:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        count += 1

        filename = f"{folder}/{name}_{count}.jpg"

        cv2.imwrite(filename, face)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        print(f"Saved: {filename}")

        cv2.waitKey(30)

        if count >= max_images:
            break

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset collection completed.")