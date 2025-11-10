from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
from pathlib import Path
import tempfile
from typing import Optional
import asyncio
from datetime import datetime

from detector import ObjectDetector
from tracker import ObjectTracker

# Initialize FastAPI app
app = FastAPI(
    title="OpenVINO Object Detection API",
    description="Real-time person-vehicle-bike detection and tracking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector and tracker instances
detector: Optional[ObjectDetector] = None
tracker: Optional[ObjectTracker] = None

# Request models
class ImageURLRequest(BaseModel):
    url: str


@app.on_event("startup")
async def startup_event():
    """Initialize detector and tracker on startup"""
    global detector, tracker
    print("Initializing OpenVINO detector...")
    detector = ObjectDetector(
        model_path="models/person-vehicle-bike-detection-crossroad-0078.xml",
        device="CPU",
        num_streams=2  # CPU_THROUGHPUT_STREAMS
    )
    tracker = ObjectTracker(method="centroid")
    print("Detector initialized successfully!")


@app.get("/")
async def root():
    """Basic info route"""
    return {
        "name": "OpenVINO Object Detection API\n",
        "version": "1.0.0\n",
        "status": "running\n",
        "endpoints\n": {
            "/upload_image/": "POST - Upload image for detection\n",
            "/image_url/": "POST - Process image from URL\n",
            "/video_stream/": "GET - Live video stream with detection\n",
            "/docs": "API documentation\n"
        },
        "model": "person-vehicle-bike-detection-crossroad-0078\n",
        "device": "CPU\n",
        "target_fps": "20-30"
    }


@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    """
    Accept image upload and return processed image with detections
    
    Args:
        file: Uploaded image file
        
    Returns:
        Processed image with bounding boxes and labels
    """
    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        detections = await detector.detect_async(image)
        
        # Draw bounding boxes
        output_image = detector.draw_detections(image, detections)
        
        # Encode image to bytes
        _, buffer = cv2.imencode('.jpg', output_image)
        
        # Return as streaming response
        return StreamingResponse(
            iter([buffer.tobytes()]),
            media_type="image/jpeg",
            headers={
                "X-Detections-Count": str(len(detections)),
                "X-Processing-Time": str(detector.last_inference_time)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/image_url/")
async def image_url(request: ImageURLRequest):
    """
    Fetch image from URL, run inference, and return processed image
    
    Args:
        request: JSON with image URL
        
    Returns:
        Processed image with bounding boxes
    """
    try:
        import urllib.request
        
        # Download image from URL
        with urllib.request.urlopen(request.url) as response:
            image_data = response.read()
        
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image URL or format")
        
        # Run detection
        detections = await detector.detect_async(image)
        
        # Draw bounding boxes
        output_image = detector.draw_detections(image, detections)
        
        # Encode image
        _, buffer = cv2.imencode('.jpg', output_image)
        
        return StreamingResponse(
            iter([buffer.tobytes()]),
            media_type="image/jpeg",
            headers={
                "X-Detections-Count": str(len(detections)),
                "X-Processing-Time": str(detector.last_inference_time)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")


from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime
import cv2, asyncio

# Global stats dictionary
object_counts = {"persons": 0, "vehicles": 0, "bikes": 0}


@app.get("/video_stream/")
async def video_stream(source: str = "0"):
    """
    Stream live annotated video with real-time detection and tracking
    
    Args:
        source: Video source (0 for webcam, file path, or URL)
        
    Returns:
        Streaming response with multipart/x-mixed-replace
    """

    async def generate_frames():
        global object_counts

        # Open video source
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video source")
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = datetime.now()
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame for performance
                frame = cv2.resize(frame, (640, 480))
                
                # Run async detection
                detections = await detector.detect_async(frame)
                
                # Update tracker
                tracked_objects = tracker.update(detections)
                
                # --- Update dashboard counts ---
                object_counts = {"persons": 0, "vehicles": 0, "bikes": 0}
                for _, obj in tracked_objects.items():
                    label = obj["label"].lower()
                    if "person" in label:
                        object_counts["persons"] += 1
                    elif "vehicle" in label:
                        object_counts["vehicles"] += 1
                    elif "bike" in label:
                        object_counts["bikes"] += 1
                # ---------------------------------
                
                # Draw tracked objects
                output_frame = detector.draw_tracked_objects(frame, tracked_objects)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 30:
                    fps_end_time = datetime.now()
                    time_diff = (fps_end_time - fps_start_time).total_seconds()
                    current_fps = fps_counter / time_diff
                    fps_counter = 0
                    fps_start_time = datetime.now()
                
                # Add FPS overlay
                cv2.putText(output_frame, f"FPS: {current_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Encode frame to JPEG
                _, buffer = cv2.imencode('.jpg', output_frame, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                # Stream frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                # Prevent CPU overload
                await asyncio.sleep(0.001)
        
        finally:
            cap.release()

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video file and return a processed version."""
    import tempfile, cv2
    import numpy as np
    import shutil

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        shutil.copyfileobj(file.file, tmp_in)
        input_path = tmp_in.name

    cap = cv2.VideoCapture(input_path)
    output_path = input_path.replace(".mp4", "_processed.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        detections = await detector.detect_async(frame)
        tracked = tracker.update(detections)
        frame = detector.draw_tracked_objects(frame, tracked)
        out.write(frame)

    cap.release()
    out.release()

    return StreamingResponse(
        open(output_path, "rb"),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename=processed_video.mp4"}
    )


from fastapi.responses import FileResponse

# Global counter dictionary
object_counts = {"persons": 0, "vehicles": 0, "bikes": 0}

@app.get("/stats")
async def get_stats():
    """
    Returns current object counts for dashboard display
    """
    return object_counts


@app.get("/dashboard")
async def get_dashboard():
    """
    Serve HTML dashboard page
    """
    return FileResponse("index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)