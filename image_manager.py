from flask import Flask, request, jsonify, make_response
from werkzeug.utils import secure_filename
from datetime import datetime
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests
import tempfile

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model_url = os.getenv("MODEL_IMAGES")
response = requests.get(model_url)
response.raise_for_status()

# Créer un fichier temporaire pour le modèle
with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
    tmp.write(response.content)
    tmp_path = tmp.name


# Charger le modèle à partir du fichier temporaire


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_filename(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_name = f"{timestamp}_{secure_filename(filename)}"
    # if ext:
    #     unique_name = f"{unique_name}"
    return unique_name


def image_error_response(message):
    return jsonify({"error": message}), 400


def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)  # Convertit l'image en tableau numpy
    img_array_expanded = np.expand_dims(img_array, axis=0)  # Ajoute une dimension pour créer un batch de taille 1
    img_preprocessed = preprocess_input(img_array_expanded)  # Prétraitement spécifique si nécessaire, dépend du modèle
    return img_preprocessed


def predict_image(img_path):
    class_indices = {'nude': 0, 'safe': 1, 'sexy': 2}
    idx_to_class = {v: k for k, v in class_indices.items()}
    model = load_model(tmp_path)
    # model = load_model(os.getenv("MODEL_IMAGES"))
    processed_image = load_and_preprocess_image(img_path)
    prediction = model.predict(processed_image)
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = idx_to_class[predicted_class_idx]
    return predicted_class, prediction[0]


def is_toxic(img_path):
    predicted_class, probabilities = predict_image(img_path)
    # print("predicted class dans is toxic est:", predicted_class)
    if predicted_class == "safe":
        return False
    else:
        return True
