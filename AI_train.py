import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.applications import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.regularizers import l2
from glob import glob

# Define constants
IMAGE_SIZE = [512, 512]  # Increased input size for better feature extraction
EPOCHS = 200
train_image_path = r"C:\Users\denys\Desktop\UI\training_images"
test_image_path = r"C:\Users\denys\Desktop\UI\test_images"
best_model_file = r'C:\Users\denys\Desktop\UI\checkpoint_ai.h5'

# Load InceptionV3 with pre-trained ImageNet weights and exclude the top layers
PretrainedModel = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Unfreeze the last 30 layers for fine-tuning
for layer in PretrainedModel.layers[:-10]: 
    layer.trainable = False
for layer in PretrainedModel.layers[-10:]:
    layer.trainable = True

# Get the number of classes by counting folders in the training directory
Classes = glob(train_image_path + "/*")
numOfClasses = len(Classes)

# Build the top layers on the InceptionV3 base model
avgPoolingLayer = GlobalAveragePooling2D()(PretrainedModel.output)
predLayer = Dense(numOfClasses, activation='softmax')(avgPoolingLayer)

# Define the final model
model = Model(inputs=PretrainedModel.input, outputs=predLayer)

# Compile the model with SGD optimizer
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    metrics=['accuracy']
)

print(model.summary())

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
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
# model.save_weights("experiment_weights.h5")
model.save("experiment_v4.h5")