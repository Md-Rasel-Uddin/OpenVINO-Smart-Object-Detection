# OpenVINO Object Detection & Tracking API

A real-time **person–vehicle–bike detection and tracking** system by **OpenVINO** and **FastAPI**.  
Includes a live web dashboard for uploading images, videos, or viewing webcam streams with annotated detections.

---

## 🧠 Features

- ✅ Real-time detection using Intel OpenVINO models  
- ✅ Detects **persons**, **vehicles**, and **bikes**  
- ✅ Object tracking with **SORT** algorithm  
- ✅ Upload images or videos directly from the dashboard  
- ✅ Supports image URLs and live webcam streaming  
- ✅ Interactive dashboard built with HTML + JavaScript  
- ✅ Ready for deployment (Vercel, Render, Railway, etc.)

---

## 📁 Project Structure

```
├── main.py                              # FastAPI entry point  
├── detector.py                          # OpenVINO model inference logic  
├── tracker.py                           # Object tracker (SORT algorithm)  
│

└── index.html                       # Frontend dashboard (HTML interface)
│
├── models/
│   ├── person-vehicle-bike-detection-crossroad-0078.xml   # OpenVINO model file
│   └── person-vehicle-bike-detection-crossroad-0078.bin   # Model weights
    └── sample video  # for testing the model 
│
├── requirements.txt                     # Python dependencies
└── README.md                            # Documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Md-Rasel-Uddin/OpenVINO-Smart-Object-Detection
cd openvino-fastapi-detection
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download the model

Place the following model files under the `/models` directory:

- `person-vehicle-bike-detection-crossroad-0078.xml`  
- `person-vehicle-bike-detection-crossroad-0078.bin`

👉 You can download them from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/person-vehicle-bike-detection-crossroad-0078).

---

## 🚀 Running the Application

### 🧩 Option 1: Using Python directly

```bash
python main.py
```

### 🧩 Option 2: Using Uvicorn (recommended)

```bash
uvicorn main:app --reload
```

Then open your browser and navigate to:

👉 **http://127.0.0.1:8000**

---

## 💡 Available Endpoints

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/` | GET | API overview and status |
| `/upload_image/` | POST | Upload an image file for detection |
| `/image_url/` | POST | Provide an image URL for remote inference |
| `/video_stream/?source=0` | GET | Live webcam stream (or file/RTSP source) |
| `/stats` | GET | Real-time detection stats (persons, vehicles, bikes) |

---

## 🌐 Dashboard Interface

The dashboard (`index.html`) lets you:
- 📸 Upload image files for detection  
- 🎥 Upload video files for processing  
- 🔗 Provide an online image URL  
- 🟢 Start live webcam video (if available)  

It automatically displays detection counts and annotated output in real time.

---

## ⚡ Performance (Example Results)

| Device | Model | Average FPS | Detected Classes |
|---------|--------|-------------|------------------|
| CPU (i7-12700H) | OpenVINO IR (FP32) | ~28 FPS | Person, Vehicle, Bike |
| Intel iGPU | OpenVINO IR (FP16) | ~40 FPS | Person, Vehicle, Bike |

🧠 The performance may vary depending on hardware, stream resolution, and concurrency.

---


## 🧾 Requirements

- Python 3.8+
- OpenVINO Runtime
- FastAPI
- Uvicorn
- OpenCV
- NumPy
- Python-Multipart

Install all at once via:

```bash
pip install -r requirements.txt
```

---

