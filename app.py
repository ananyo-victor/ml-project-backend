from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import base64
import io

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load your pre-trained model
try:
    model = load_model("facial_expression_model.keras")
except:
    print("model not loaded")
if model:
    print("model loaded")

# Define classes (update these based on your model's output)
classes = ["Happy", "Sad", "Angry"]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the Base64 frame from the frontend
        data = request.json.get("frame")
        if not data:
            return jsonify({"error": "No frame received"}), 400

        # Decode the Base64 image
        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Convert to grayscale using OpenCV
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize the image to 48x48 and normalize the pixel values
        image = cv2.resize(image, (48, 48))
        image = image.astype('float32') / 255.0

        # Expand the dimensions to match the model input (1, 48, 48, 1)
        image_array = np.expand_dims(image, axis=-1)  # Add channel dimension
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(image_array)
        predicted_class = classes[np.argmax(predictions)]

        # Send the predicted emotion back to the frontend
        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
