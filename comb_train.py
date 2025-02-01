# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras import backend as K
import os

# Custom F1-score metric as a class
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Custom callback to log precision, recall, and F1-score
class MetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}:")
        print(f"  Training Accuracy: {logs.get('accuracy'):.4f}")
        print(f"  Validation Accuracy: {logs.get('val_accuracy'):.4f}")
        print(f"  Training Precision: {logs.get('precision'):.4f}")
        print(f"  Validation Precision: {logs.get('val_precision'):.4f}")
        print(f"  Training Recall: {logs.get('recall'):.4f}")
        print(f"  Validation Recall: {logs.get('val_recall'):.4f}")
        print(f"  Training F1-Score: {logs.get('f1_score'):.4f}")
        print(f"  Validation F1-Score: {logs.get('val_f1_score'):.4f}")

# Constants for hyperparameters
IMG_SIZE = 512  # Image size for InceptionV3
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
NUM_CLASSES = 120  # Number of dog breeds
TRAIN_DATA_DIR = r'/home/jovyan/AI-dog-identifier-project/combined_datasets/combined_datasets/training_images'
TEST_DATA_DIR = r'/home/jovyan/AI-dog-identifier-project/combined_datasets/combined_datasets/test_images'
CHECKPOINT_PATH = 'model_checkpoints/best_model_unfreeze30.h5'

# Data augmentation only for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Reserve 20% for validation
)

# Minimal preprocessing for validation set
val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2  # Same split as above
)

# Generators for training and validation data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
    subset='training'  # Use the training subset
)

val_generator = val_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # Use the validation subset
)

# Generator for testing data
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # No shuffling for test data
)

# Load InceptionV3 base model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze all layers of the base model except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling
x = BatchNormalization()(x)  # Batch normalization
x = Dropout(DROPOUT_RATE)(x)  # Dropout layer for regularization
predictions = Dense(NUM_CLASSES, activation='softmax')(x)  # Prediction layer

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        F1Score(name='f1_score')  # Use class-based F1-score metric
    ]
)

# Callbacks
if not os.path.exists('model_checkpoints'):
    os.makedirs('model_checkpoints')

checkpoint = ModelCheckpoint(CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
metrics_logger = MetricsLogger()  # Custom callback

callbacks = [checkpoint, early_stopping, reduce_lr, metrics_logger]

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# Evaluate the model on the test set
test_loss, test_accuracy, test_precision, test_recall, test_f1 = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1-Score: {test_f1}")

# Save the final model
model.save('dog_breed_classifier_final_unfreeze20.h5')
