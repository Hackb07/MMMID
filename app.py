import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model1 = YOLO("brain-tumor.pt")
model2 = YOLO("breasr-cancer.pt")
model3 = YOLO("bone-fracture.pt")


# Define custom class names for each model
custom_classes_model1 = {0: "tumor"}  # Example for Model 1
custom_classes_model2 = {0: "cancer"}  # Example for Model 2
custom_classes_model3 = {0: "fracture"}  # Example for Model 3

st.title("Multi-Model Medical Image Detection")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Brain-Tumor", "Breast-Cancer", "Bone-Fracture"])

def process_image(model, uploaded_image, class_dict):
    image = Image.open(uploaded_image)
    results = model(image)  # Run inference
    
    # Get detected class names and confidence scores
    detected_classes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get class index
            class_name = class_dict.get(class_id, f"Unknown({class_id})")  # Use custom class names
            confidence = box.conf[0].item()  # Get confidence score
            detected_classes.append(f"{class_name} ({confidence:.2f})")
    
    img_with_boxes = results[0].plot()  # Get annotated image
    return Image.fromarray(img_with_boxes), detected_classes

# Tab 1 - Model 1
with tab1:
    st.header("Multi-Model Medical Image Detection(Brain-Tumor)")
    uploaded_file1 = st.file_uploader("Upload an Image for Brain-Tumor", type=["jpg", "png", "jpeg"], key="file1")
    if uploaded_file1:
        output_img1, detected_classes1 = process_image(model1, uploaded_file1, custom_classes_model1)
        st.image(output_img1, caption="Processed Image (Model 1)", use_column_width=True)
        if detected_classes1:
            st.subheader("Detected Objects:")
            for obj in detected_classes1:
                st.write(f"✅ {obj}")

# Tab 2 - Model 2
with tab2:
    st.header("Multi-Model Medical Image Detection(Breast-Cancer)")
    uploaded_file2 = st.file_uploader("Upload an Image for Breast-Cancer", type=["jpg", "png", "jpeg"], key="file2")
    if uploaded_file2:
        output_img2, detected_classes2 = process_image(model2, uploaded_file2, custom_classes_model2)
        st.image(output_img2, caption="Processed Image (Model 2)", use_column_width=True)
        if detected_classes2:
            st.subheader("Detected Objects:")
            for obj in detected_classes2:
                st.write(f"✅ {obj}")

# Tab 3 - Model 3
with tab3:
    st.header("Multi-Model Medical Image Detection(Bone-Fracture)")
    uploaded_file3 = st.file_uploader("Upload an Image for Bone-Fracture", type=["jpg", "png", "jpeg"], key="file3")
    if uploaded_file3:
        output_img3, detected_classes3 = process_image(model3, uploaded_file3, custom_classes_model3)
        st.image(output_img3, caption="Processed Image (Model 3)", use_column_width=True)
        if detected_classes3:
            st.subheader("Detected Objects:")
            for obj in detected_classes3:
                st.write(f"✅ {obj}")
