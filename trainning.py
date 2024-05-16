import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# Load image data
normal_cells = os.listdir('dataset/no')
tumor_cells = os.listdir('dataset/yes')
normal_path = 'dataset/no/'
tumor_path = 'dataset/yes/'
data = []

for img_file in normal_cells:
    image = Image.open(normal_path + img_file)
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

for img_file in tumor_cells:
    image = Image.open(tumor_path + img_file)
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

# Convert data and labels to numpy arrays
X = np.array(data)
normal_label = [0] * len(normal_cells)
tumor_label = [1] * len(tumor_cells)
Y = np.array(normal_label + tumor_label)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

# Preprocess data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build and train the model
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, Y_train, epochs=5, validation_split=0.1, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print("Test Accuracy:", accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.legend()
plt.show()

# Save the model
model.save('tumor_detection.keras')
