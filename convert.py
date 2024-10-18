import tensorflow as tf
from tensorflow.keras.models import load_model #type:ignore

# Load the .keras model
model_path = 'emotion.keras'
model = load_model(model_path)

# Save the model in .h5 format
model.save('emotion_model.h5')
