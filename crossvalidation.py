import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical, load_img, img_to_array
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
from glob import glob

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Reduce batch size
BATCH_SIZE = 16
IMAGE_SIZE = [331, 331]  # Reduced resolution to save memory
NUM_CLASSES = 120  # Replace with the actual number of classes
K_FOLDS = 5

# Prepare dataset
train_image_path = r"C:\Users\denys\Desktop\UI\training_images"
image_paths = []
labels = []

for class_id, class_name in enumerate(sorted(os.listdir(train_image_path))):
    class_folder = os.path.join(train_image_path, class_name)
    for img_file in glob(class_folder + "/*"):
        image_paths.append(img_file)
        labels.append(class_id)

image_paths = np.array(image_paths)
labels = np.array(labels)

# Load images in batches
def load_images(image_paths, target_size):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images)

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
    print(f"Training on Fold {fold + 1}/{K_FOLDS}...")

    # Load train and validation data
    train_images = load_images(image_paths[train_idx], target_size=IMAGE_SIZE)
    val_images = load_images(image_paths[val_idx], target_size=IMAGE_SIZE)
    train_labels = to_categorical(labels[train_idx], num_classes=NUM_CLASSES)
    val_labels = to_categorical(labels[val_idx], num_classes=NUM_CLASSES)

    # Load the saved model
    model_path = r"C:\Users\denys\Desktop\UI\fine_tune_model_v4.h5"
    model = load_model(model_path)

    # Train the model
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        batch_size=BATCH_SIZE,
        epochs=10,  # Fewer epochs for cross-validation
        verbose=1
    )

    # Evaluate validation accuracy
    val_accuracy = max(history.history['val_accuracy'])
    print(f"Fold {fold + 1} Validation Accuracy: {val_accuracy:.4f}")
    fold_accuracies.append(val_accuracy)

# Compute and display cross-validation results
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"Cross-Validation Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
