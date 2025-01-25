import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st



# Get the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for the model and class indices
model_path = os.path.join(working_dir, "trained_model", "DSML_PROJECT.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load the pre-trained model
try:
    model = tf.keras.models.load_model(model_path)
except FileNotFoundError:
    st.error("Error: Model file not found. Please ensure 'DSML_PROJECT.h5' exists in the 'trained_model' directory.")
    st.stop()

# Load class indices
try:
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
except FileNotFoundError:
    st.error("Error: Class indices file not found. Please ensure 'class_indices.json' exists in the project directory.")
    st.stop()

# Function to preprocess the image
def load_and_preprocess_image(image, target_size=(224, 224)):
    # Convert uploaded file to an image
    img = Image.open(image)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize image data to [0, 1]
    img_array = img_array.astype("float32") / 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit app
st.title("Plant Disease Classifier")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        # Display uploaded image
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image")

    with col2:
        if st.button("Classify"):
            try:
                # Predict the class of the uploaded image
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f"Prediction: {prediction}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
