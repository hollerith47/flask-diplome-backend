from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import numpy as np
import requests
import tempfile
import os

load_dotenv()


# model = load_model(os.getenv("MODEL_TEXT"))

def get_model():
    model_url = os.getenv("MODEL_TEXT")
    response = requests.get(model_url)
    response.raise_for_status()

    # Créer un fichier temporaire pour le modèle
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    # Charger le modèle à partir du fichier temporaire
    model = load_model(tmp_path)
    return model


# Charger le tokenizer
def get_tokenizer():
    tokenizer_url = os.getenv("TOKENIZER_URL")
    response = requests.get(tokenizer_url)
    response.raise_for_status()
    tokenizer_data = response.text
    tokenizer = tokenizer_from_json(tokenizer_data)

    return tokenizer


# with open('tokenizer.json') as f:
#     data = f.read()
#     tokenizer = tokenizer_from_json(data)


def prepare_text(texts, max_length=120):
    tokenizer = get_tokenizer()
    if isinstance(texts, str):
        texts = [texts]
    sequences = tokenizer.texts_to_sequences(texts)
    # print("Séquences :", sequences)  # Ajoutez cette ligne pour le débogage
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded


def predict_toxicity(texts):
    model = get_model()
    prepared_texts = prepare_text(texts)
    predictions = model.predict(prepared_texts)
    return (predictions > 0.5).astype(np.int32)
