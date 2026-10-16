# Brain Tumor Detection & Segmentation Dashboard

An end-to-end medical imaging web application designed to detect brain tumors from MRI/CT scans using **YOLO11**, generate precise anatomical segmentations using **Segment Anything Model 2 (SAM2)**, and compute standardized clinical metrics via an interactive **Streamlit** dashboard. Fully containerized and published on Docker Hub for instant cross-platform deployment.

---

## 🚀 Key Features

- **Dual-Stage AI Pipeline:** 
  * **YOLO11 Detection:** Quickly localizes and classifies brain tumors into categories (`meningioma`, `pituitary`, `glioma`, etc.) while filtering out `NO_tumor` scans to prevent unnecessary masking.
  * **SAM2 Segmentation:** Generates fine-grained pixel masks exclusively over validated tumor bounding boxes.
- **Clinical Readout & Metrics:** Automatically computes standardized medical dimensions including **Cross-Sectional Area** ($cm^2$ and $mm^2$) and **Maximum Linear Dimension** ($cm$ and $mm$) using dynamic pixel spacing calibration.
- **Interactive Controls:** Features real-time sliders for adjusting YOLO confidence thresholds and pixel spacing physical calibration scales.
- **Containerized Infrastructure:** Packed with a lightweight, optimized `Dockerfile` and hosted on Docker Hub for seamless deployment across any server environment without dependency friction.

---

## 🛠️ Tech Stack

- **Frontend & UI:** Streamlit
- **Object Detection:** Ultralytics YOLO11 (`best.pt`)
- **Segmentation:** Ultralytics SAM2 (`sam2_b.pt`)
- **Image Processing:** OpenCV, Pillow, NumPy
- **Data Structuring:** Pandas, PyTorch (CPU-optimized)
- **Containerization:** Docker & Docker Hub

---

## 📦 Quick Start via Docker Hub (Recommended)

You can run the pre-built container image instantly without having to clone the source code or download separate model weights:

1. **Pull and run the container:**
   ```bash
   docker run -p 8501:8501 adarshch2006/brain-tumor-app:latest

## 📂 Project Directory Structure

```text
tumor_detectionProject/
│
├── weights/
│   ├── best.pt         # Trained YOLO11 tumor detection weights
│   └── sam2_b.pt       # SAM2 base model weights for segmentation
│
├── app.py              # Main Streamlit dashboard script
├── requirements.txt    # Python dependencies list
├── Dockerfile          # Container configuration file
└── README.md           # Project documentation