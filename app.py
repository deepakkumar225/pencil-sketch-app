import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Title
st.title("🎨 Pencil Sketch Generator")
st.write("Upload an image and convert it into a pencil sketch!")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image with PIL
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Convert to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Invert the gray image
    inverted_image = cv2.bitwise_not(gray_image)

    # Blur the inverted image
    blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)

    # Invert the blurred image
    inverted_blur = cv2.bitwise_not(blurred)

    # Create the sketch
    sketch = cv2.divide(gray_image, inverted_blur, scale=256.0)

    # Show results
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    st.subheader("Pencil Sketch")
    st.image(sketch, use_column_width=True, clamp=True)

    # Save option
    result = Image.fromarray(sketch)
    st.download_button(
        label="Download Sketch",
        data=result.tobytes(),
        file_name="sketch_output.jpg",
        mime="image/jpeg"
    )
