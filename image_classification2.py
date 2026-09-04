from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pretrained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Image path
image_path = "images/dog.jpg"

# Load image and resize it
img = image.load_img(image_path, target_size=(224, 224))

# Convert image to NumPy array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Preprocess image
img_array = preprocess_input(img_array)

# Make prediction
predictions = model.predict(img_array)

# Decode prediction
results = decode_predictions(predictions, top=5)[0]

print("\nPrediction Results:")
print("-------------------")

for imagenet_id, label, probability in results:
    print(f"{label}: {probability * 100:.2f}%")