from flask import Flask, request, jsonify
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
import time
import pyrebase

app = Flask(__name__)

# Load the trained image processing model
model_image_processing = tf.keras.models.load_model('finasih.h5')

# Firebase configuration
config = {
    "apiKey": "AIzaSyBPOYDA0ttZvGglnXKpwSCNqodThovupSM",
    "authDomain": "springjal-66c38.firebaseapp.com",
    "projectId": "springjal-66c38",
    "storageBucket": "springjal-66c38.appspot.com",
    "messagingSenderId": "989140358334",
    "appId": "1:989140358334:web:7c58efd5fd43e957297aef",
    "measurementId": "G-2XPYW8VGBG",
    "serviceAccount": "serviceAccount.json",
    "databaseURL": "https://springjal-66c38-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()

@app.route('/')
def index():
    return "<center><h1>Flask App Process Image </h1></center>"

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image URL from the request
        image_url = request.json['image_url']

        # Download the image from Firebase Storage
        image_response = requests.get(image_url)
        image = Image.open(BytesIO(image_response.content)).convert('L')
        image = image.resize((256, 256))
        input_array = np.array(image) / 255.0
        input_array = np.expand_dims(input_array, axis=-1)
        input_array = np.expand_dims(input_array, axis=0)

        # Make predictions
        predicted_mask = model_image_processing.predict(input_array)

        # Post-process the predicted mask
        threshold = 0.4
        binary_mask = (predicted_mask > threshold).astype(np.uint8)

        # Convert the binary mask array to a PIL Image
        output_image = Image.fromarray(binary_mask[0, ..., 0] * 255)

        # Use the current timestamp to create a unique identifier
        timestamp = int(time.time())

        # Save the output image to BytesIO in JPEG format
        output_image_bytesio = BytesIO()
        output_image.save(output_image_bytesio, format='JPEG')
        output_image_bytesio.seek(0)

        # Save the BytesIO object to Firebase Storage with a unique path
        output_image_path = f'output_images/output_{timestamp}.jpg'
        storage.child(output_image_path).put(output_image_bytesio.getvalue())

        return jsonify({'output_image_url': storage.child(output_image_path).get_url(None)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__=="__main__":
    app.run()
