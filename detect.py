import os
import cv2  # Ensure OpenCV is imported for image processing
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D #type:ignore
from tensorflow.keras.utils import to_categorical #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #type:ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Paths to the training and testing datasets
train_data_path = "D:\\Projects\\Emotion Detection\\archive (1)\\train"  # Path to the training images
test_data_path = "D:\\Projects\\Emotion Detection\\archive (1)\\test"    # Path to the testing images

def load_images_from_folder(folder):
    X = []
    y = []
    emotion_map = {  # Map folder names to integer labels
        'angry': 0,
        'happy': 1,
        'sad': 2,
        'surprised': 3,
        'neutral': 4,
        'disgust': 5,
        'fear': 6
    }
    
    for emotion in os.listdir(folder):
        if emotion in emotion_map:  # Check if the folder name is in the mapping
            emotion_path = os.path.join(folder, emotion)
            if os.path.isdir(emotion_path):  # Check if it's a directory
                for img_name in os.listdir(emotion_path):
                    img_path = os.path.join(emotion_path, img_name)
                    # Load the image
                    img = cv2.imread(img_path)
                    # Resize the image to 48x48 pixels
                    img = cv2.resize(img, (48, 48))
                    # Convert to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Normalize the image
                    img = img / 255.0
                    # Append the image and label
                    X.append(img)
                    y.append(emotion_map[emotion])  # Use the mapped label
    return np.array(X), np.array(y)

# Load training and testing datasets
X_train, y_train = load_images_from_folder(train_data_path)
X_test, y_test = load_images_from_folder(test_data_path)

# Reshape the data into 48x48 grayscale images
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)


model = Sequential()
model.add(Input(shape=(48, 48, 1)))

# First Convolutional Block
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Block
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Block
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth Convolutional Block
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Use Global Average Pooling instead of Flatten
model.add(GlobalAveragePooling2D())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))  # Dropout for regularization
model.add(Dense(7, activation='softmax')) # Output layer (7 emotion classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

filepath='D:\\Projects\\Emotion Detection\\model\\best_model.keras'  

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test),
          callbacks=[early_stopping, model_checkpoint], verbose=1)  


model_save_path = 'D:\\Projects\\Emotion Detection\\model\\emotion.keras'
model.save(model_save_path)

print(f"Model saved to {model_save_path}")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Angry', 'Happy', 'Sad', 'Surprised', 'Neutral', 'Disgust', 'Fear'],
            yticklabels=['Angry', 'Happy', 'Sad', 'Surprised', 'Neutral', 'Disgust', 'Fear'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# Function to display sample predictions
def display_sample_predictions(X, y_true, y_pred, num_samples=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[i].reshape(48, 48), cmap='gray')
        plt.title(f'True: {np.argmax(y_true[i])}, Pred: {np.argmax(y_pred[i])}')
        plt.axis('off')
    plt.show()

# Display sample predictions
display_sample_predictions(X_test, y_test, y_pred)

