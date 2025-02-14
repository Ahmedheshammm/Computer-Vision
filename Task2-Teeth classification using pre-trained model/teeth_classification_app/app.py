import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Page config
st.set_page_config(
    page_title="Dental Classification",
    page_icon="🦷",
    layout="wide"
)

# Constants
IMG_SIZE = 224
CLASS_NAMES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']


# Model loading
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('teeth_classification_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array / 255.0


# Main app
st.title("Dental Image Classification")
st.write("Upload a dental image for classification")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Classify button
    if st.button('Classify'):
        with st.spinner('Processing...'):
            try:
                model = load_model()
                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img)

                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = float(np.max(prediction))

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.success(f"Predicted Class: {predicted_class}")
                    st.info(f"Confidence: {confidence:.2%}")

                with col2:
                    # Probability distribution
                    st.write("Class Probabilities")
                    probs_df = pd.DataFrame({
                        'Class': CLASS_NAMES,
                        'Probability': prediction[0] * 100
                    })
                    st.bar_chart(probs_df.set_index('Class'))

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")