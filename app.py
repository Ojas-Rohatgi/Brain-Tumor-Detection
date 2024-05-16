from flask import Flask, render_template, request
import cv2
import tensorflow as tf
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = tf.keras.models.load_model('tumor_detection.keras')


# Function to load and preprocess image
def preprocess_image(image_path):
    input_image = cv2.imread(image_path)
    if input_image is None:
        print("Error: Unable to read the image.")
        return None
    input_image_resized = cv2.resize(input_image, (128, 128))
    image_normalized = input_image_resized / 255.0
    img_reshape = image_normalized.reshape((1, 128, 128, 3))
    return img_reshape


# Function to make predictions
def make_predictions(model, img_reshape):
    input_prediction = model.predict(img_reshape)
    input_pred_label = 'Tumor Cell' if input_prediction > 0.5 else 'Normal Cell'
    return input_pred_label


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            img_reshape = preprocess_image(file_path)
            if img_reshape is not None:
                prediction = make_predictions(model, img_reshape)
                return render_template('result.html', prediction=prediction, filename=file.filename)
            else:
                return render_template('index.html', error="Error processing the image.")
    return render_template('index.html', error="Something went wrong. Please try again.")


if __name__ == '__main__':
    app.run(debug=True)