import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.utils import load_img, img_to_array
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from keras import backend as K
from run_nn import F1Score  # Используем F1Score из run_nn.py

# Отключение GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Ограничение использования памяти GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled for GPU.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")

# Настройки модели
IMAGE_SIZE = 512
train_image_path = r"/home/jovyan/AI-dog-identifier-project/combined_datasets/combined_datasets/training_images"
CLASSES = sorted(os.listdir(train_image_path))  # Сортируем для корректной индексации
num_classes = len(CLASSES)

best_model_file = r'/home/jovyan/AI-dog-identifier-project/model_checkpoints/best_model_unfreeze30.h5'
custom_objects = {"F1Score": F1Score}
model = load_model(best_model_file, custom_objects=custom_objects)

# Функция для подготовки изображения
def prepare_img(image_path):
    image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Создание сервера Flask
app = Flask(__name__)

# Маршрут для обработки изображений
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Сохранение изображения во временный файл
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    # Подготовка изображения и предсказание
    try:
        img_for_model = prepare_img(temp_image_path)
        result_array = model.predict(img_for_model, verbose=0)
        confidence = float(np.max(result_array))  # Максимальная вероятность
        answer = np.argmax(result_array, axis=1)
        index = answer[0]
        class_name = CLASSES[index]
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Удаление временного изображения
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    return jsonify({'class_name': class_name, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
