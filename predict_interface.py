import tensorflow as tf
import numpy as np
from PIL import Image
from pyimagesearch.simplecnn import SimpleCNN

# Load your trained model
model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)
model.load_weights('model_weights.weights.h5')  # Update with the actual path to your model weights


# Load the image
image = Image.open("image3.jpeg")

# Resize the image to 28x28 pixels
resized_image = image.resize((28, 28))

# Convert the image to grayscale (if necessary)
grayscale_image = resized_image.convert("L")

# Convert the image to a numpy array
image_array = np.array(grayscale_image)

# Normalize the pixel values to be between 0 and 1
normalized_image = image_array / 255.0

# Expand the dimensions to match the input shape expected by the model
input_image = np.expand_dims(normalized_image, axis=0)

# Define function to preprocess input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((28, 28))  # Adjust dimensions as needed
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert to RGB if not already in RGB mode
    image = np.array(image) / 255.0  # Normalize pixel values
    if len(image.shape) == 2:  # If grayscale, add channel dimension
        image = np.expand_dims(image, axis=-1)
    return np.expand_dims(image, axis=0)  # Add batch dimension


# Define function to run inference
def run_inference(image):
    prediction = model.predict(image)
    return prediction

# Function to get user input and display results
def main():
    image_path = input("Enter path to the image: ")
    # input_image = preprocess_image(image_path)
    prediction = run_inference(input_image)
    print("Predicted class probabilities:", prediction)

if __name__ == "__main__":
    main()