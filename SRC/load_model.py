# here we are testing to load the model which was created in the previous 
# step inside the train_model.py

# src/load_model.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model("models/mnist_model.h5")

# Test a random digit (use a sample from the dataset)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0  # Normalize
sample = x_test[0].reshape(1, 28, 28)

# Predict
predicted_class = np.argmax(model.predict(sample))
print(f"Predicted class: {predicted_class}")