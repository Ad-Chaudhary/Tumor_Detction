import streamlit as st
from ultralytics import YOLO, SAM
from PIL import Image
import numpy as np
import cv2
import torch
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide"
)

# 2. Helper Function for Comprehensive Medical Measurements (cm & mm)
def calculate_medical_tumor_metrics(sam_result, pixel_spacing_mm=0.2):
    """
    Calculates standard clinical tumor metrics in both cm and mm units.
    """
    if sam_result.masks is None or len(sam_result.masks) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    
    masks = sam_result.masks.data.cpu().numpy()
    combined_mask = np.any(masks, axis=0)
    pixel_count = np.sum(combined_mask)
    
    # Area calculations (cm² and mm²)
    area_mm2 = pixel_count * (pixel_spacing_mm ** 2)
    area_cm2 = area_mm2 / 100.0
    
    # Linear dimension calculations (cm and mm)
    y_indices, x_indices = np.where(combined_mask)
    if len(y_indices) > 0 and len(x_indices) > 0:
        max_height_pixels = np.max(y_indices) - np.min(y_indices)
        max_width_pixels = np.max(x_indices) - np.min(x_indices)
        max_span_pixels = max(max_height_pixels, max_width_pixels)
        max_diameter_mm = max_span_pixels * pixel_spacing_mm
        max_diameter_cm = max_diameter_mm / 10.0
    else:
        max_diameter_mm = 0.0
        max_diameter_cm = 0.0

    return float(area_cm2), float(area_mm2), float(max_diameter_cm), float(max_diameter_mm), int(pixel_count)

# 3. Model Loading (Cached for performance on local CPU)
@st.cache_resource
def load_ai_models():
    device = "cpu"
    
    # Using relative paths for deployment portability
    yolo_path = "weights/best.pt"
    sam_path = "weights/sam2_b.pt"
    
    yolo_model = YOLO(yolo_path)
    sam_model = SAM(sam_path)
    
    return yolo_model, sam_model, device

with st.spinner("Loading AI Models into memory... Please wait."):
    yolo_model, sam_model, device = load_ai_models()

# 4. Streamlit Sidebar Layout & Performance Insights
st.sidebar.header("Model Controls")
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
pixel_spacing = st.sidebar.slider("Pixel Spacing Scale (mm/pixel)", 0.05, 0.5, 0.2, 0.01, 
                                  help="Adjusts physical area scaling calibration.")

st.sidebar.markdown("---")
st.sidebar.subheader("Validation Insights")
st.sidebar.info(
    "**Note on Classes:**\n"
    "- High Accuracy: `NO_tumor`, `meningioma`\n"
    "- Moderate: `pituitary`, `glioma`\n"
    "- *Warning:* `space-occupying lesion-` has sparse training samples."
)

# 5. Main UI Header
st.title("Brain Tumor Detection & Segmentation Dashboard")
st.markdown("Upload a brain scan image to detect tumors using your trained **YOLO11** weights and generate physical segmentations using **SAM2**.")

uploaded_file = st.file_uploader("Choose a brain scan image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Scan")
        st.image(image, width="stretch")

    if st.button("Run Detection & Segmentation", type="primary"):
        with st.spinner("Processing image through YOLO11 & SAM2 (CPU)..."):
            
            # Step 1: Run YOLO Detection
            yolo_results = yolo_model(image_np, conf=conf_threshold, device=device, verbose=False)
            
            detected_boxes = []
            class_names_found = []
            confidences = []
            
            for result in yolo_results:
                boxes = result.boxes
                if len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    class_ids = boxes.cls.int().tolist()
                    conf_scores = boxes.conf.cpu().numpy()
                    
                    # FILTER: Ignore 'NO_tumor' class boxes so they don't trigger SAM2 masking
                    for box, cid, conf in zip(xyxy, class_ids, conf_scores):
                        class_name = yolo_model.names[cid]
                        if class_name.lower() != "no_tumor":
                            detected_boxes.append(box)
                            class_names_found.append(class_name)
                            confidences.append(conf)

            with col2:
                st.subheader("Segmentation Results")
                
                if len(detected_boxes) == 0:
                    st.success("No tumors detected! Scan is clear.")
                    st.image(image, width="stretch")
                else:
                    # Step 2: Run SAM2 Segmentation ONLY on valid tumor boxes
                    sam_results = sam_model(image_np, bboxes=np.array(detected_boxes), device=device, verbose=False)
                    
                    # Step 3: Calculate Dual-Unit Medical Metrics
                    area_cm2, area_mm2, max_dia_cm, max_dia_mm, tumor_pixels = calculate_medical_tumor_metrics(
                        sam_results[0], pixel_spacing_mm=pixel_spacing
                    )
                    
                    # Step 4: Render Mask Overlay Image
                    segmented_image_np = sam_results[0].plot() 
                    segmented_image_rgb = cv2.cvtColor(segmented_image_np, cv2.COLOR_BGR2RGB)
                    
                    st.image(segmented_image_rgb, width="stretch")
                    st.success(f"Successfully segmented {len(detected_boxes)} tumor region(s)!")

                    # Step 5: Clinical Readout Display
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric(label="Cross-Sectional Area", value=f"{area_cm2:.2f} cm²", delta=f"{area_mm2:.1f} mm²")
                    with metric_col2:
                        st.metric(label="Max Linear Dimension", value=f"{max_dia_cm:.2f} cm", delta=f"{max_dia_mm:.1f} mm")

                    # Step 6: Detailed Detection breakdown table
                    st.markdown("### Detected Class Breakdown")
                    breakdown_data = []
                    for name, conf in zip(class_names_found, confidences):
                        breakdown_data.append({
                            "Class Prediction": name, 
                            "Confidence Score": f"{conf * 100:.2f}%"
                        })
                    st.table(pd.DataFrame(breakdown_data))