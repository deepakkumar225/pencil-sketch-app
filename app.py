import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Title
st.title("🎨 Pencil Sketch Generator")
st.write("Upload an image and compare Original vs Pencil Sketch side by side!")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image with PIL
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    # ✅ Resize image for faster processing
    max_width, max_height = 800, 800
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Convert to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Invert gray
    inverted_image = cv2.bitwise_not(gray_image)

    # Blur
    blurred = cv2.GaussianBlur(inverted_image, (15, 15), 0)

    # Invert blur
    inverted_blur = cv2.bitwise_not(blurred)

    # Sketch
    sketch = cv2.divide(gray_image, inverted_blur, scale=256.0)

    # Add Laplacian edges for darker strokes
    edges = cv2.Laplacian(gray_image, cv2.CV_8U, ksize=5)
    edges = cv2.bitwise_not(edges)
    dark_sketch = cv2.addWeighted(sketch, 0.6, edges, 0.4, 0)

    # 🟢 Show results in two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("✏️ Pencil Sketch")
        st.image(dark_sketch, use_column_width=True, clamp=True)
