import sys
import numpy as np

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.preprocessing import image


if len(sys.argv) < 2:
    print("Please provide an image path.")
    print("Example: python image_classification.py images/TwoWhiller.jpg")
    sys.exit()


image_path = sys.argv[1]

model = MobileNetV2(weights="imagenet")

img = image.load_img(image_path, target_size=(224, 224))

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = preprocess_input(img_array)

prediction = model.predict(img_array)

result = decode_predictions(prediction, top=1)

label = result[0][0][1]

confidence = result[0][0][2] * 100


print("\nImage Classification Result")
print("----------------------------")
print("Image:", image_path)
print("Prediction:", label)
print("Confidence: {:.2f}%".format(confidence))