import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load trained model
MODEL_PATH = "trained_plant_disease_model.keras"  # Ensure this is the correct path
model = tf.keras.models.load_model(MODEL_PATH)

# Define validation dataset directory
VALID_DIR = r"E:\New Plant Diseases Dataset(Augmented)\valid"


# Load class names from dataset
validation_set = tf.keras.utils.image_dataset_from_directory(
    directory=VALID_DIR, labels="inferred", label_mode="categorical", batch_size=32,
    image_size=(128, 128), shuffle=True
)
class_names = validation_set.class_names  # Extract class labels

# Streamlit UI
st.title("🌿 Plant Disease Detection App")
st.write("Upload an image of a plant leaf, and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Convert file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Display uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image for model
    image_resized = cv2.resize(img, (128, 128))  # Resize to match model input
    input_arr = img_to_array(image_resized)
    input_arr = np.array([input_arr]) / 255.0  # Normalize
    
    # Predict disease
    st.write("Predicting disease...")
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)  # Get highest probability index
    
    # Debugging Statements
    print("Class Names:", class_names)
    print("Result Index:", result_index)
    
    # Ensure index is within bounds
    if result_index < len(class_names):
        predicted_class = class_names[result_index]
        st.success(f"🌱 Predicted Disease: **{predicted_class}**")
    else:
        st.error("❌ Error: Prediction index out of range. Check model output!")