import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
import cv2
import seaborn as sns
from shutil import copyfile
from PIL import Image

from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.utils import layer_utils
from keras import backend as K
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D

labels = pd.read_csv('C:/Users/denys/Desktop/UI/dog-breed-identification/labels.csv')

labels_dict = {i:j for i,j in zip(labels['id'],labels['breed'])}
classes = set(labels_dict.values())
images = [f for f in os.listdir('C:/Users/denys/Desktop/UI/dog-breed-identification/train')]

if  not os.path.exists('training_images'):
        os.makedirs('training_images')

if  not os.path.exists('test_images'):
    os.makedirs('test_images')
    
os.chdir('training_images')
for curClass in classes:    
    if  not os.path.exists(curClass):
        os.makedirs(curClass)
        #os.rmdir(curClass)

os.chdir('../test_images')
for curClass in classes:    
    if  not os.path.exists(curClass):
        os.makedirs(curClass)
        


os.chdir('..')
count = 0 
destination_directory = 'training_images/'
for item in images:
    if count >7999:
        destination_directory = 'test_images/'
    filekey = os.path.splitext(item)[0]
    if  not os.path.exists(destination_directory+labels_dict[filekey]+'/'+item):
        copyfile('C:/Users/denys/Desktop/UI/dog-breed-identification/train/'+item, destination_directory+labels_dict[filekey]+'/'+item)
    # print(labels_dict[filekey])
    count +=1

img = Image.open(r'C:\Users\denys\Desktop\UI\training_images\afghan_hound\0d5a88f0ab2db8d34b533c69768135e8.jpg')
img.show()

datagen = ImageDataGenerator(
    rotation_range=50,
    width_shift_range=0.3,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

img = Image.open(r'C:\Users\denys\Desktop\UI\training_images\afghan_hound\0d5a88f0ab2db8d34b533c69768135e8.jpg')

# Convert the PIL image to a NumPy array
x = np.array(img)

# Reshape to add batch dimension
x = x.reshape((1,) + x.shape)

# Ensure the save directory exists
save_dir = r'C:\Users\denys\Desktop\UI\preview'
os.makedirs(save_dir, exist_ok=True)

# Generate and save augmented images
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=save_dir, save_prefix='dog_breed', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # Stop after generating 20 images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'training_images',
        target_size=(128, 128),
        batch_size=20,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'test_images',
        target_size=(128, 128),
        batch_size=20,
        class_mode='categorical')