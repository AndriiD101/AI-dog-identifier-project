import tensorflow as tf
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


IMAGE_SIZE = 224
train_image_path = r"C:\Users\denys\Desktop\UI\training_images"
CLASSES = os.listdir(train_image_path)
num_classes = len(CLASSES)

best_model_file = r'C:\Users\denys\Desktop\UI\experiment.h5'
model = tf.keras.models.load_model(best_model_file)

def prepare_img(path_for_image):
    # Load the image with the target size
    image = load_img(path_for_image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    # Convert the image to an array
    img_result = img_to_array(image)
    # Expand dimensions to match model input shape and normalize
    img_result = np.expand_dims(img_result, axis=0)
    img_result = img_result / 255.0
    return img_result

test_img_path = r"C:\Users\denys\Desktop\UI\something.jpg"

# Prepare the image for model prediction
img_for_model = prepare_img(test_img_path)

# Predict using the loaded model
result_array = model.predict(img_for_model, verbose=1)
print(result_array)

answer = np.argmax(result_array, axis=1)
print(answer)

index = answer[0]

class_name = CLASSES[index]
print(class_name)