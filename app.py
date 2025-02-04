import cv2
import os
import json
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection
)
import os
import tempfile

# -------------------- Configuration -------------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_EXTRACTION_INTERVAL = 0.01  # Seconds between frame captures

# -------------------- Model Loading -------------------- #
try:
    print("🔄 Loading visual model and processor...")
    processor_visual = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model_visual = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(DEVICE)
    print(f"✅ Model loaded on {DEVICE} successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# -------------------- Metadata Extraction -------------------- #
def extract_metadata(video_path):
    """Extracts video metadata using FFmpeg"""
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
               "-show_format", "-show_streams", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
    except Exception as e:
        print(f"❌ Metadata extraction failed: {e}")
        return {}

# -------------------- Frame Extraction -------------------- #
def extract_frames(video_path, output_folder="frames"):
    """Extracts frames from video at specified interval (supports sub-second intervals)"""
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video file")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video
    total_duration = total_frames / fps  # Total duration in seconds
    frame_count = 0

    # Use a while loop for sub-second intervals
    timestamp = 0.0
    while timestamp <= total_duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)  # Convert seconds to milliseconds
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.jpg", frame)
            frame_count += 1
        else:
            break  # Stop if we can't read any more frames
        
        timestamp += FRAME_EXTRACTION_INTERVAL  # Increment by the interval
    
    cap.release()
    return frame_count
# -------------------- Optical Flow Calculation -------------------- #
def calculate_optical_flow(frames_folder):
    """Calculates dense optical flow between consecutive frames with validation"""
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])
    flow_results = []
    
    # Get reference dimensions from first valid frame
    ref_height, ref_width = None, None
    for f in frame_files:
        frame = cv2.imread(os.path.join(frames_folder, f))
        if frame is not None:
            ref_height, ref_width = frame.shape[:2]
            break

    if ref_height is None:
        print("⚠ No valid frames found for optical flow calculation")
        return []

    prev_gray = None
    for i in tqdm(range(len(frame_files)), desc="Calculating optical flow"):
        current_path = os.path.join(frames_folder, frame_files[i])
        current_frame = cv2.imread(current_path)

        if current_frame is None:
            continue
            
        # Ensure consistent dimensions
        if current_frame.shape[:2] != (ref_height, ref_width):
            current_frame = cv2.resize(current_frame, (ref_width, ref_height))

        # Ensure 3-channel color format
        if len(current_frame.shape) == 2:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, current_gray, None, 
                pyr_scale=0.5, levels=3, iterations=3,
                winsize=15, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            flow_magnitude = np.sqrt(flow[...,0]*2 + flow[...,1]*2)
            flow_results.append({
                "max_flow": float(flow_magnitude.max()),
                "mean_flow": float(flow_magnitude.mean())
            })

        prev_gray = current_gray

    # Apply temporal smoothing
    window_size = 5
    smoothed_flow = []
    for i in range(len(flow_results)):
        start = max(0, i - window_size // 2)
        end = min(len(flow_results), i + window_size // 2 + 1)
        window = flow_results[start:end]
        avg_mean = np.mean([f['mean_flow'] for f in window])
        avg_max = np.mean([f['max_flow'] for f in window])
        smoothed_flow.append({'mean_flow': avg_mean, 'max_flow': avg_max})
    
    return smoothed_flow

# -------------------- Visual Analysis -------------------- #
def detect_objects(frames_folder):
    """Processes frames through the visual detection model"""
    results = []
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])
    
    for frame_file in tqdm(frame_files, desc="Analyzing frames"):
        try:
            image = Image.open(os.path.join(frames_folder, frame_file))
            inputs = processor_visual(images=image, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                outputs = model_visual(**inputs)

            # Process detections with lower threshold
            target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)
            detections = processor_visual.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.4  # Lowered from 0.7
            )[0]

            scores = detections["scores"].cpu().numpy().tolist()
            max_confidence = max(scores) if scores else 0.0
            
            results.append({
                "frame": frame_file,
                "detections": len(scores),
                "max_confidence": max_confidence,
                "average_confidence": np.mean(scores) if scores else 0.0
            })
            
        except Exception as e:
            print(f"⚠ Error processing {frame_file}: {e}")
            results.append({
                "frame": frame_file,
                "detections": 0,
                "max_confidence": 0.0,
                "average_confidence": 0.0
            })
    
    return results

# -------------------- Manipulation Detection -------------------- #
def detect_manipulation(report_path="report.json"):
    """Determines video authenticity based on analysis results"""
    try:
        with open(report_path) as f:
            report = json.load(f)
            
        # Adjusted thresholds
        CONFIDENCE_THRESHOLD = 0.80  # Reduced from 0.65
        FLOW_STD_THRESHOLD = 28      # New standard deviation threshold
        SUSPICIOUS_FRAME_RATIO = 0.3  # Increased from 0.25
        
        stats = report["summary_stats"]
        
        # New metrics
        confidence_std = np.std([r["average_confidence"] for r in report["frame_analysis"]])
        flow_std = stats.get("std_optical_flow", 0)
        low_conf_frames = sum(1 for r in report["frame_analysis"] if r["average_confidence"] < 0.4)
        anomaly_ratio = low_conf_frames / len(report["frame_analysis"])

        # Multi-factor scoring
        score = 0
        if stats["average_detection_confidence"] < CONFIDENCE_THRESHOLD:
            score += 1.5
        if flow_std > FLOW_STD_THRESHOLD:
            score += 1.2
        if anomaly_ratio > SUSPICIOUS_FRAME_RATIO:
            score += 1.0
        if confidence_std > 0.2:  # High variance in confidence
            score += 0.8

        return score

    except Exception as e:
        return f"❌ Error in analysis: {str(e)}"

# -------------------- Reporting -------------------- #
# -------------------- Reporting -------------------- #
def generate_report(visual_results, flow_results, output_file="report.json"):
    """Generates comprehensive analysis report"""
    report_data = {
        "frame_analysis": visual_results,
        "motion_analysis": flow_results,
        "summary_stats": {
            "max_detection_confidence": max(r["max_confidence"] for r in visual_results),
            "average_detection_confidence": np.mean([r["average_confidence"] for r in visual_results]),
            "detection_confidence_std": np.std([r["average_confidence"] for r in visual_results]),
            "peak_optical_flow": max(r["max_flow"] for r in flow_results) if flow_results else 0,
            "average_optical_flow": np.mean([r["mean_flow"] for r in flow_results]) if flow_results else 0,
            "std_optical_flow": np.std([r["mean_flow"] for r in flow_results]) if flow_results else 0
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(report_data, f, indent=2)

    # ... rest of visualization code ...
    
    return report_data  # Added return statement

# -------------------- Main Pipeline -------------------- #
def analyze_video(video_path):
    """Complete video analysis workflow"""
    print("\n📋 Metadata Extraction:")
    metadata = extract_metadata(video_path)
    print(json.dumps(metadata.get("streams", [{}])[0], indent=2))

    print("\n🎞 Frame Extraction:")
    frame_count = extract_frames(video_path)
    print(f"✅ Extracted {frame_count} frames at {FRAME_EXTRACTION_INTERVAL}s intervals")

    print("\n🔍 Running object detection...")
    visual_results = detect_objects("frames")

    print("\n🌀 Calculating optical flow...")
    flow_results = calculate_optical_flow("frames")

    print("\n📊 Generating Final Report...")
    report_data = generate_report(visual_results, flow_results)

    print("\n🔍 Authenticity Analysis:")
    score = detect_manipulation()  # This function should return a score

    print(f"\n🎯 Final Score: {score}")  # Debugging line
    return score  # ✅ Ensure this score is returned properly



# -------------------- Execution -------------------- #



#--------------------------------Streamlit---------------------------------------------#
#--------------------------------Streamlit---------------------------------------------#
import streamlit as st
import tempfile
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")  # Ensure you have a separate style.css file

# Sidebar for Navigation
# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Home", "Analyze Video", "About"])

# Home Page
if page == "Home":
    st.markdown("<h1 class='title'>Video Manipulation Detection</h1>", unsafe_allow_html=True)
    
    # Hero Section
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='hero-text'>
            Detect manipulated videos with AI-powered analysis.
            Protect yourself from deepfakes and synthetic media.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.video("Realistic Universe Intro_free.mp4")  # Add sample video URL
    
    # Features Section
    st.markdown("## How It Works")
    cols = st.columns(3)
    with cols[0]:
        st.image("upload-icon.png", width=100)
        st.markdown("### Upload Video")
    with cols[1]:
        st.image("analyze-icon.png", width=100)
        st.markdown("### AI Analysis")
    with cols[2]:
        st.image("result-icon.png", width=100)
        st.markdown("### Get Results")


elif page == "Analyze Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name  # ✅ Correct variable name

        st.video(temp_video_path)

        if st.button("Analyze Video"):
            with st.spinner("Analyzing..."):
                try:
                    score = analyze_video(temp_video_path)  # ✅ Ensure function exists
                    
                    # Debugging Line
                    st.write(f"Analysis Score: {score}")  
                    float(score)
                    # Display result based on score
                    if score >= 3.5 :
                        st.markdown(f"""
                        <div class='result-box suspicious'>
                            <p>This video shows major signs of manipulation</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif score >= 2.0:
                        st.markdown(f"""
                        <div class='result-box suspicious'>
                            <p>This video shows minor signs of manipulation</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-box clean'>
                            <p>No significant manipulation detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

elif page == "About":  # ✅ Now this will work correctly
    st.markdown("<h1 class='title'>About Us</h1>", unsafe_allow_html=True)
    
    # Creator Profile
    col1, col2 = st.columns(2)
    with col1:
        st.image("creator.jpg", width=300, caption="Ayush Agarwal, Lead Developer")
    with col2:
        st.markdown("""
        <div class='about-text'>
            ## Ayush Agarwal ,
            Student at VIT Bhopal University ,
            AIML enthusiast 
            <br><br>
            📧 ayush.23bce10678@vitbhopal.ac.in
            <br>
            🔗 [LinkedIn](www.linkedin.com/in/ayush20039939)
            <br>
            🐙 [GitHub](https://github.com)
        </div>
        """, unsafe_allow_html=True)
    
    # Technology Stack
    st.markdown("## Our Technology")
    st.markdown("""
    <div class='tech-stack'>
        <img src='https://img.icons8.com/color/96/000000/python.png'/>
        <img src='https://img.icons8.com/color/96/000000/tensorflow.png'/>
        <img src='https://img.icons8.com/color/96/000000/opencv.png'/>
        <img src='https://raw.githubusercontent.com/github/explore/968d1eb8fb6b704c6be917f0000283face4f33ee/topics/streamlit/streamlit.png'/>
    </div>
    """, unsafe_allow_html=True)
