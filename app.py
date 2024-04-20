from flask import Flask, request, jsonify, make_response
from text_manager import predict_toxicity
from image_manager import image_error_response, allowed_file, unique_filename, is_toxic

from dotenv import load_dotenv
import os

load_dotenv()
UPLOAD_FOLDER = "./uploads"

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return 'Welcome to flask Backend'


@app.route("/get")
def create():
    # texts = ["Это хороший день!", "Я тебя ненавижу, и ты ужасен."]
    texts = "Я тебя ненавижу, и ты ужасен."
    predictions = predict_toxicity(texts)
    # print(predictions[0])
    response = {"texte": texts, "toxicity": True if predictions[0][0] else False}
    print(response)
    return 'Hello World! yes'


# @app.route("/find", methods=['POST'])
# def find():
#     data = request.json
# value = request.form['name'] # pour le form
# fichier = request.files['nom_du_fichier'] # pour le fichier
# texts = ["Это хороший день!", "Я тебя ненавижу, и ты ужасен."]
# predictions = predict_toxicity(texts)
# for text, pred in zip(texts, predictions):
# print("text", text)
# print("pred", pred[0])
# print(f"Texte: {text}\nToxique: {'Oui' if pred[0] else 'Non'}\n")
# value = {"toxicity": predict_toxicity(data["texte"])}
# response = make_response(jsonify(value), 200)
# return response
@app.post("/api/check-images")
def check_images():
    if 'images' not in request.files:
        image_error_response("Aucune image fournie")
    image = request.files.get("image")

    if image.filename == "":
        image_error_response("le fichier fourni n'est pas de format image")

    if image and allowed_file(image.filename):
        filename = unique_filename(image.filename)
        # save the image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)

        # Traitement de l'image ici (comme la prédiction du modèle)
        try:
            result = is_toxic(file_path)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        return jsonify({"is_toxic": result, "image": filename})

    return image_error_response("Type de fichier non autorisé")


@app.route("/api/check-texts", methods=['POST'])
def check_texts():
    data = request.json
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "Aucun texte fourni"}), 400
    predictions = predict_toxicity(texts)
    # print(predictions[0])
    response = [{"texte": texts, "is_toxic": True if predictions[0][0] else False}]

    return jsonify(response), 200


if __name__ == '__main__':
    # // development
    # app.run(debug=True, port=5500)
    # // production
    app.run(debug=False, host="0.0.0.0")
