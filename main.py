import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# File paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load model and class indices
model = tf.keras.models.load_model(model_path)
with open(class_indices_path, 'r') as json_file:
    class_indices = json.load(json_file)

# Helper functions
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f4f4f9;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #1abc9c;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌱 Plant Disease Classifier")
st.markdown(
    "Upload an image of a plant leaf, and the AI model will predict the disease or confirm if it's healthy. Let's keep our plants happy and thriving! 🌿"
)

uploaded_image = st.file_uploader("Upload an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Uploaded Image:")
        resized_img = Image.open(uploaded_image).resize((150, 150))
        st.image(resized_img, use_container_width=False)

    with col2:
        st.markdown("### Classification Result:")
        if st.button('🔍 Classify Now'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"**Prediction:** {prediction}")
            st.balloons()  # Add animation for a delightful experience
        else:
            st.info("Click the button to classify the uploaded image.")

# Footer
st.markdown(
    """
    ---
    Developed with ❤️ by Ahalya.
    """,
    unsafe_allow_html=True,
)
