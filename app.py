import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation, MaxPooling2D,
                                     Dropout, GlobalAveragePooling2D, Dense)
from tensorflow.keras.initializers import HeNormal

# === CONFIG ===
INPUT_SHAPE = (32, 32, 3)
WEIGHTS_PATH = "best_model.weights.h5"  # path to your saved weights file

CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

# === MODEL DEFINITION ===
def traffic_sign_net(input_shape):
    initializer = HeNormal()
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=initializer, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Classification
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    return model

# === LOAD MODEL WITH WEIGHTS ===
@st.cache_resource
def load_model():
    model = traffic_sign_net(INPUT_SHAPE)
    model.load_weights(WEIGHTS_PATH)
    return model

model = load_model()

# === PREDICTION FUNCTION ===
def predict_traffic_sign(image: Image.Image):
    # Convert image to RGB (3 channels), in case input is grayscale or RGBA
    image = image.convert('RGB')
    
    # Resize to model expected input shape (width, height)
    image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Now shape should be (32, 32, 3)
    # Add batch dimension
    img_array = img_array.reshape(1, *INPUT_SHAPE)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    return CLASS_NAMES[predicted_index], confidence

# === STREAMLIT APP ===
st.title("🚦 Traffic Sign Recognition")
st.write("Upload an image of a traffic sign and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        label, confidence = predict_traffic_sign(image)

    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")
