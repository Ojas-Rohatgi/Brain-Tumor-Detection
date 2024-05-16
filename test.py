import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


model = tf.keras.models.load_model('tumor_detection_02.keras')


# Function to load and preprocess image
def preprocess_image(image_path):
    input_image = cv2.imread(image_path)
    if input_image is None:
        print("Error: Unable to read the image.")
        return None
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.show()

    # Resize image
    input_image_resized = cv2.resize(input_image, (128, 128))

    # Normalize pixel values
    image_normalized = input_image_resized / 255.0

    # Reshape for model prediction
    img_reshape = image_normalized.reshape((1, 128, 128, 3))

    return img_reshape


# Function to make predictions
def make_predictions(model, img_reshape):
    # Make predictions
    input_prediction = model.predict(img_reshape)

    # Get the predicted label
    input_pred_label = 'Tumor Cell' if input_prediction > 0.5 else 'Normal Cell'
    print('Predicted Label:', input_pred_label)

    # Display prediction probabilities
    print('Prediction Probabilities are:', input_prediction)


if __name__ == "__main__":
    # Input image path
    # input_image_path = input('Enter the path of the image: ')

    # input_image = cv2.imread(input_image_path)
    # if input_image is None:
    #     print("Error: Unable to read the image.")

    # Preprocess image
    # img_reshape = preprocess_image("img_1.png")         #Normal Cell
    img_reshape = preprocess_image("sample.jpeg")       #Tumor Cell
    # Make predictions
    make_predictions(model, img_reshape)


