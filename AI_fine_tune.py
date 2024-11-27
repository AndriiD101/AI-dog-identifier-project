import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from glob import glob

# Define constants
IMAGE_SIZE = [331, 331]  # Increased input size for better feature extraction
EPOCHS = 200
train_image_path = r"C:\Users\denys\Desktop\UI\training_images"
test_image_path = r"C:\Users\denys\Desktop\UI\test_images"
best_model_file = r'C:\Users\denys\Desktop\UI\fine_tune_exp_v4.h5'

# Load your pre-trained model (replace 'your_model_path.h5' with the actual path to your model)
model = load_model(r'C:\Users\denys\Desktop\UI\models\experiment_v4.h5')

# Freeze the layers you don't want to train (adjust the layers as needed)
# Let's freeze all layers except the last 50 layers for fine-tuning
for layer in model.layers[:-50]:  # Freeze layers before the last 50
    layer.trainable = False
for layer in model.layers[-50:]:  # Unfreeze the last 50 layers for fine-tuning
    layer.trainable = True

# Get the number of classes by counting folders in the training directory
Classes = glob(train_image_path + "/*")
numOfClasses = len(Classes)

# Compile the model with a lower learning rate for fine-tuning
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.0001, momentum=0.09),  # Lower learning rate for fine-tuning
    metrics=['accuracy']
)

print(model.summary())

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.4,   # Increase zoom range to focus more on breed-specific features
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.8, 1.4],
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data
training_set = train_datagen.flow_from_directory(
    train_image_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_image_path,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# Define callbacks
callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)
]

# Train the model
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=EPOCHS,
    callbacks=callbacks,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Print the best validation accuracy
best_val_acc = max(r.history['val_accuracy'])
print(f"Best validation accuracy: {best_val_acc}")

# Save the model weights and the full model
model.save("fine_tune_model_v4.h5")
