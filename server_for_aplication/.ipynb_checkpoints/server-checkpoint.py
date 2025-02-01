import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Настройки модели
IMAGE_SIZE = 512
train_image_path = r"C:\Users\ssssm\PycharmProjects\Server\dogdogup\training_images"
CLASSES = os.listdir(train_image_path)
num_classes = len(CLASSES)

best_model_file = r'C:\Users\ssssm\PycharmProjects\Server\dogdogup\best_model_unfreeze30.h5'
model = tf.keras.models.load_model(best_model_file)

# Функция для подготовки изображения к предсказанию
def prepare_img(image_path):
    image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result, axis=0)
    img_result = img_result / 255.0
    return img_result

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
    app.run(debug=True, host='0.0.0.0', port=5000)
