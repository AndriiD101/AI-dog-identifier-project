import os
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.metrics import Metric
from tensorflow.keras import backend as K

# Define the custom metric (F1 Score)
class F1Score(Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.round(y_pred)
        y_true = K.cast(y_true, "float32")
        y_pred = K.cast(y_pred, "float32")
        
        tp = K.sum(y_true * y_pred)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        return 2 * (precision * recall) / (precision + recall + K.epsilon())

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

# Register the custom object
custom_objects = {"F1Score": F1Score}

# Load the trained model
model = load_model(
    r'/home/jovyan/AI-dog-identifier-project/model_checkpoints/best_model_unfreeze30.h5', 
    custom_objects=custom_objects
)

# Path to the folder containing subfolders of dog breeds (test dataset)
test_folder = r'/home/jovyan/AI-dog-identifier-project/combined_datasets/combined_datasets/test_images'

# List of all dog breeds (folder names) in the test dataset
dog_breeds = sorted(os.listdir(test_folder))
print(dog_breeds)

# Define the confidence threshold
CONFIDENCE_THRESHOLD = 0.7

# Initialize counters
total_images = 0
true_certain = 0
false_certain = 0
true_uncertain = 0
false_uncertain = 0

# Iterate through each breed folder
for breed in dog_breeds:
    breed_folder_path = os.path.join(test_folder, breed)
    
    # Ensure it's a folder
    if os.path.isdir(breed_folder_path):
        # Iterate through each image in the folder
        for image_name in os.listdir(breed_folder_path):
            image_path = os.path.join(breed_folder_path, image_name)
            
            # Ensure it's a file
            if os.path.isfile(image_path):
                # Load and preprocess the image
                img = load_img(image_path, target_size=(512, 512))  # Resize for InceptionV3
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                img_array = preprocess_input(img_array)  # Preprocess for InceptionV3

                # Predict the dog breed
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions)  # Get the index of the highest score
                confidence = predictions[0][predicted_class]  # Get the confidence of the top prediction
                predicted_breed = dog_breeds[predicted_class]

                # Check the prediction outcome
                if confidence < CONFIDENCE_THRESHOLD:
                    # Uncertain prediction
                    if predicted_breed == breed:
                        false_uncertain += 1  # Correct prediction marked as uncertain
                    else:
                        true_uncertain += 1  # Incorrect prediction marked as uncertain
                else:
                    # Certain prediction
                    if predicted_breed == breed:
                        true_certain += 1  # Correct confident prediction
                    else:
                        false_certain += 1  # Incorrect confident prediction

                total_images += 1

# Calculate accuracy for confident predictions
confident_predictions = true_certain + false_certain
accuracy = (true_certain / confident_predictions) * 100 if confident_predictions > 0 else 0

# Print detailed report
print(f"Total images: {total_images}")
print(f"Confident predictions: {confident_predictions}")
print(f"Uncertain predictions: {true_uncertain + false_uncertain}")
print(f"True Certain Predictions: {true_certain}")
print(f"False Certain Predictions: {false_certain}")
print(f"True Uncertain Predictions: {true_uncertain}")
print(f"False Uncertain Predictions: {false_uncertain}")
print(f"Confident Prediction Accuracy: {accuracy:.2f}%")

