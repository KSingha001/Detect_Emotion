import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model and face detector
model = load_model('models/emotion_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels (adjust if dataset has different order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to find available cameras
def find_camera():
    """Try to find working camera, prioritizing Camo Studio"""
    print("Searching for available cameras...")
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera found at index {i}")
                return cap
            cap.release()
    return None

# Start webcam (Camo Studio or any available camera)
cap = find_camera()

if cap is None:
    print("❌ Error: No camera found. Make sure Camo Studio is running.")
    print("Steps to fix:")
    print("1. Open Camo Studio app on your phone and computer")
    print("2. Connect your phone to computer (USB or WiFi)")
    print("3. Make sure Camo Studio virtual camera is enabled")
    exit()

print("Starting emotion detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float')/255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)[0]
        label = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f'{label} ({confidence:.1f}%)', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Emotion Detector - Press Q to Quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Emotion detection stopped.")