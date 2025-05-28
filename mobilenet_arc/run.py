

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('mangrove_model_mobilenet.h5')

# Image to predict
img_path = 'sample.png'  # Ganti dengan path gambar uji
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
class_idx = np.argmax(pred)

# Label mapping (ganti sesuai kelas aslinya)
class_labels = ['Avicennia', 'Bruguiera', 'Ceriops', 'Rhizophora', 'Sonneratia']
print("Predicted class:", class_labels[class_idx])
