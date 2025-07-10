import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Import preprocessing functions
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from keras.applications.inception_v3 import preprocess_input as inception_preprocess

# Class names
CLASS_NAMES = [
    'bass', 'black_sea_sprat', 'gilt_head_bream', 'hourse_mackerel',
    'red_mullet', 'red_sea_bream', 'sea_bass', 'shrimp',
    'striped_red_mullet', 'trout'
]

st.set_page_config(page_title="🐟 Fish Species Classifier", layout="centered")
st.title("🐟 Fish Species Classifier")
st.markdown("Upload a fish image and select a model to predict its species.")

# Upload image
uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

# Model selection
model_options = {
    "EfficientNetB0": r"D:\image_classification\model\efficientnetb0_fish_finetuned.h5",
    "MobileNet": r"D:\image_classification\model\mobilenet_fish_finetuned.h5",
    "InceptionV3": r"D:\image_classification\model\inceptionv3_fish_finetuned.h5",
    "ResNet50": r"D:\image_classification\model\resnet50_fish_finetuned.h5",
    "VGG16": r"D:\image_classification\model\vgg16_fish_finetuned.h5",
    "CNN from Scratch": r"D:\image_classification\model\fish_cnn_model.h5"
}
selected_model = st.selectbox("Select a model", list(model_options.keys()))

# Show image preview
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

# Show Submit button only when both are selected
if uploaded_file and selected_model:
    submit = st.button("🔍 Submit for Prediction")

    if submit:
        # Resize and convert to array
        image = image.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)

        # Choose preprocess function
        if "EfficientNet" in selected_model:
            preprocess_func = efficientnet_preprocess
        elif "ResNet" in selected_model:
            preprocess_func = resnet_preprocess
        elif "VGG" in selected_model:
            preprocess_func = vgg_preprocess
        elif "MobileNet" in selected_model:
            preprocess_func = mobilenet_preprocess
        elif "Inception" in selected_model:
            preprocess_func = inception_preprocess
        else:
            preprocess_func = lambda x: x  # No preprocessing for custom CNN

        # Apply preprocessing
        img_array = preprocess_func(img_array)

        # Load model and predict
        model_path = model_options[selected_model]
        model = tf.keras.models.load_model(model_path)

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_label = CLASS_NAMES[predicted_index]

        # Display result
        st.success(f"🧠 Predicted Species: **{predicted_label}**")
        st.markdown("🔢 Confidence Scores:")
        st.bar_chart({cls: float(score) for cls, score in zip(CLASS_NAMES, predictions)})
