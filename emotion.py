import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings

import cv2
import numpy as np
from cv2 import data as cv2_data # type: ignore
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

# Load the pre-trained emotion detection model
model_path = 'D:\\Projects\\Emotion Detection\\Emotion-Detection\\model\\emotion.keras'
model = load_model(model_path)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam or video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2_data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Preprocess the face for emotion detection
        face = gray_frame[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))  # Resize to the input size of the model
        face = face.astype('float32') / 255.0  # Normalize the pixel values
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict the emotion
        predictions = model.predict(face)
        emotion_index = np.argmax(predictions[0])
        emotion = emotion_labels[emotion_index]

        # Display the emotion label on the frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Show the frame with detected faces and emotions
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
