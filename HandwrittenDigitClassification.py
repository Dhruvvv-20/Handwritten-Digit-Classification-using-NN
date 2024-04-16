import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the pixel values
X_train = X_train / 255
X_test = X_test / 255

# Reshape the data
X_train_flattened = X_train.reshape(len(X_train), 28 * 28)
X_test_flattened = X_test.reshape(len(X_test), 28 * 28)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(X_train_flattened, y_train, epochs=5)

# Evaluate the model
model.evaluate(X_test_flattened, y_test)

# Generate a random index
random_index = np.random.randint(0, len(X_test))

# Select a random image from the test dataset
random_image = X_test_flattened[random_index]

# Reshape the image to the required input shape
random_image_reshaped = random_image.reshape(1, 28 * 28)

# Make a prediction
predicted_number = model.predict(random_image_reshaped)

# Output the predicted number
print("The predicted number is:", np.argmax(predicted_number))

# Display the random image
plt.imshow(X_test[random_index], cmap='gray')
plt.axis('off')
plt.show()
