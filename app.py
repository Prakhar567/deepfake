import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to detect deepfake in a video
def detect_video_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (224, 224)))  # Assuming the model takes input size (224, 224))
    cap.release()

    frames = np.array(frames)
    frames = frames / 255.

    # Example: Simple rule-based detection based on frame analysis
    # You would need a more sophisticated method for real deepfake detection
    if len(frames) > 1000:  # Example: If the video is longer than 1000 frames, classify as real
        return "Real😎 you can trust it"
    else:
        return "Deepfake☠️ Be safe"

# Function to detect deepfake in an image using computer vision techniques
def detect_image_deepfake(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Calculate variance of Laplacian
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Example threshold for variance
    threshold = 100
    if variance < threshold:
        return "Deepfake☠️ Be safe"  # If variance is low, likely a deepfake
    else:
        return "Real😎 you can trust it"  # If variance is high, likely not a deepfake

# Streamlit app setup
st.title("Deepfake AI Detection App")

# Tabs for image and video
tab1, tab2 = st.tabs(["Image", "Video"])

# Image Tab
with tab1:
    st.subheader("Detect Deepfake in Image")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the image file
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform deepfake detection on the image
        result = detect_image_deepfake(image)
        st.write("Detection Result:", result)
    else:
        st.write("Please upload an image.")

# Video Tab
with tab2:
    st.subheader("Detect Deepfake in Video")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        # Display the video
        st.video("temp_video.mp4")

        # Perform deepfake detection on the video
        result = detect_video_deepfake("temp_video.mp4")
        st.write("Detection Result:", result)
    else:
        st.write("Please upload a video.")
