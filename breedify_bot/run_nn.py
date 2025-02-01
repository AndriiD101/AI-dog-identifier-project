import os
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.metrics import Metric
from keras import backend as K

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

# Define the function for single image prediction
def predict_single_image(image_path, model_path, label_path, confidence_threshold=0.7):
    """
    Predict the breed of a single dog image using a trained model.

    Args:
        image_path (str): Path to the image file.
        model_path (str): Path to the trained model file.
        label_path (str): Path to the folder containing subfolders of dog breeds.
        confidence_threshold (float): Threshold to determine if prediction is confident.

    Returns:
        dict: Prediction details (breed, confidence, certainty).
    """
    try:
        # Load the trained model
        model = load_model(model_path, custom_objects=custom_objects)

        # List of all dog breeds (folder names)
        dog_breeds = sorted(os.listdir(label_path))

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

        # Determine if prediction is confident or uncertain
        certainty = "Certain" if confidence >= confidence_threshold else "Uncertain"

        # Return prediction details
        return {
            "Predicted Breed": predicted_breed,
            "Confidence": confidence,
            "Certainty": certainty
        }

    except Exception as e:
        return {"Error": str(e)}

# Example usage
# model_path = r"C:\Users\denys\Desktop\UI\telegram bot\best_model_unfreeze30_test.h5"
# image_path = r"C:\Users\denys\Desktop\UI\1Dog-rough-collie-portrait.jpg"
# label_path = r"C:\Users\denys\Desktop\UI\test_images"

# result = predict_single_image(image_path, model_path, label_path)
# print(result)
