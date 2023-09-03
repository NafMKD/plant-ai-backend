from flask import Flask, request, jsonify
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import json
import numpy as np
import cv2
import os
import requests
import openai

app = Flask(__name__)

openai.api_key = "sk-738888409673629799-83730"
openai.api_base = "https://api.shuttle.rip/v1"

# Load the TensorFlow model
model_url = "https://tfhub.dev/rishit-dagli/plant-disease/1"
model = hub.load(model_url)

# Define a function to preprocess the image
def preprocess_image(img):
    # Resize the image to 224x224 pixels
    target_size = (224, 224)
    img = image.img_to_array(img)
    img = cv2.resize(img, target_size)

    # Normalize color values to the range [0, 1]
    img = img / 255.0

    return img[np.newaxis, ...]

@app.route("/predict", methods=["POST"])
def predict_image():
    try:
        # Check if the JSON file exists
        json_filename = 'class_indices.json'
        if not os.path.exists(json_filename):
            # Download the JSON file if it doesn't exist
            json_url = "https://github.com/Rishit-dagli/Greenathon-Plant-AI/releases/download/v0.1.0/class_indices.json"
            response = requests.get(json_url)
            if response.status_code == 200:
                with open(json_filename, 'wb') as json_file:
                    json_file.write(response.content)

        # Load the class map JSON file
        with open(json_filename, 'r') as json_file:
            class_map = json.load(json_file)

        # Read and preprocess the uploaded image
        img_file = request.files['image']
        img_file.save('image.jpg')

        img = image.load_img('image.jpg')
        img = preprocess_image(img)

        # Make predictions
        predictions = model(img)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class label from the class map
        predicted_class_label = class_map.get(str(predicted_class_index), "Unknown")

        return jsonify({"predicted_class_label": predicted_class_label, "probability": str(predictions.numpy()[0][predicted_class_index])})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/gpt", methods=["POST"])
def generativeAI():
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {
                'role': 'user', 
                'content': request.get_json()["message"]
            },
        ]
    )

    return response

if __name__ == "__main__":
    app.run(debug=True)
