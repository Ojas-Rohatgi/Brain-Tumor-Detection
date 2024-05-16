# Brain Tumor Detection

This project aims to detect brain tumors from MRI scans using a Convolutional Neural Network (CNN) model. The application includes a training script for the model, a Flask web application for image upload and prediction, and necessary configuration files for Docker.

## Directory Structure
```plaintext
BrainTumorDetection/
│
├── dataset/
│   ├── no/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── yes/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── static/
│   └── uploads/
│       └── (uploaded images)
│
├── templates/
│   ├── index.html
│   ├── result.html
│
├── app.py
├── training.py
├── Dockerfile
└── README.md
```
## Dataset Description
The dataset consists of MRI scan images categorized into two folders:
- `no`: Contains images of normal brain cells.
- `yes`: Contains images of brain cells with tumors.

Each image is resized to 128x128 pixels and converted to RGB format for processing.

## Training the Model

The `training.py` script is used to train a Convolutional Neural Network (CNN) to detect brain tumors from the MRI scans.

### Steps:
1. Load and preprocess image data from the `dataset/no` and `dataset/yes` directories.
2. Split the data into training and testing sets.
3. Normalize the image data.
4. Build and compile the CNN model.
5. Train the model on the training set and validate it.
6. Evaluate the model on the test set.
7. Save the trained model to `tumor_detection_02.keras`.

### Usage:
```bash
python training.py
```

## Flask Web Application

The `app.py` script sets up a Flask web application that allows users to upload an MRI image and receive a prediction on whether the image shows a normal cell or a tumor cell.

### Routes:
- `/`: Home page with an image upload form.
- `/predict`: Handles image upload, processes the image, and returns the prediction result.

### Usage:
```bash
flask run
```

### Running with Docker
1. Build the Docker image:
    ```bash
    docker build -t brain-tumor-detection .
    ```

2. Run the Docker container:
    ```bash
    docker run -p 5000:5000 brain-tumor-detection
    ```

## HTML Templates

- `index.html`: Home page template with an image upload form.
- `result.html`: Result page template displaying the prediction result.

## Dockerfile

The Dockerfile sets up the environment to run the Flask application:
1. Uses the official Python 3.10 slim image.
2. Copies the project files into the container.
3. Installs the necessary Python packages.
4. Exposes port 5000.
5. Runs the Flask application.

## Requirements

- Flask
- TensorFlow
- OpenCV
- PIL
- NumPy
- Matplotlib
- scikit-learn

Install the dependencies using `pip`:
```bash
pip install flask tensorflow opencv-python-headless pillow numpy matplotlib scikit-learn
```

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/username/repo.git
    cd repo
    ```

2. **Set up the virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the training script**:
    ```bash
    python training.py
    ```

5. **Run the Flask application**:
    ```bash
    flask run
    ```

6. **Access the web application** at `http://127.0.0.1:5000/` and upload an MRI image to get the prediction.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
