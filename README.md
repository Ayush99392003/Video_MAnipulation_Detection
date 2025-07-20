
# 🎥 AI-Powered Video Manipulation Detection System

This project is an AI-based video analysis pipeline built using **PyTorch**, **Hugging Face Transformers**, **OpenCV**, and **Streamlit**, designed to detect signs of manipulation in videos. It provides insights into object inconsistencies and motion anomalies using object detection and optical flow.

---

## 🚀 Features

- 🔍 **Object Detection** (via Facebook DETR - ResNet50)
- 🌀 **Optical Flow Analysis** for motion irregularities
- 📊 **Scoring System** for authenticity estimation
- 📁 **FFmpeg Metadata Extraction**
- 🎛 **Streamlit UI** for interactive use

---

## 🧠 Workflow

1. **Upload Video** through the Streamlit UI.
2. **Frame Extraction** every 0.05 seconds.
3. **Object Detection** on each frame.
4. **Optical Flow Calculation** between frames.
5. **Generate Report** with confidence, flow, and anomaly metrics.
6. **Final Score** to indicate authenticity.

---

## 📁 Directory Structure

```
.
├── frames/                 # Extracted frames
├── style.css              # Custom Streamlit styles
├── creator.jpg            # Developer image
├── icons/                 # Optional: UI icons
├── main.py                # Core Streamlit application
├── report.json            # Auto-generated analysis report
└── README.md              # This file
```

---

## 🛠 Dependencies

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**Required Libraries:**

- `opencv-python`
- `torch`
- `transformers`
- `Pillow`
- `matplotlib`
- `numpy`
- `tqdm`
- `streamlit`

You also need **FFmpeg** installed and available in your system `PATH`.

---

## 🧪 Run Locally

```bash
streamlit run main.py
```

---

## 📈 Scoring System

- **Score ≥ 3.5**: Major manipulation likely
- **Score ≥ 2.0 and < 3.5**: Minor signs of tampering
- **Score < 2.0**: No significant manipulation detected

---

## 👨‍💻 Developer

**Ayush Agarwal**  
Student, VIT Bhopal University  
Email: ayush.23bce10678@vitbhopal.ac.in  
[LinkedIn](https://www.linkedin.com/in/ayush20039939) | [GitHub](https://github.com/Ayush99392003)

---

## 🧠 Acknowledgements

- Facebook AI - DETR Object Detection
- HuggingFace Transformers
- Streamlit Team

---

*Built with ❤️ for research and learning.*
