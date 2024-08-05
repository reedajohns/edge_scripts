import cv2
from typing import Union, Dict, Any
import os
import uvicorn
import threading
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import subprocess
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline, BufferFillingStrategy, BufferConsumptionStrategy
from inference.core.interfaces.camera.entities import VideoFrame
import docker
import io
import base64
from PIL import Image
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import numpy as np
# Import Camera Frame Grabbers
from LucidFrameProducer import LucidFrameProducer
from BaslerFrameProducer import BaslerFrameProducer

try:
    #silence annoying supervision warnings from InferencePipeline
    import warnings
    from supervision.utils.internal import SupervisionWarnings # type: ignore
    warnings.filterwarnings('ignore', category=SupervisionWarnings)
except:
    #this doesn't work when debugging for some reason, just ignore
    pass

# Latest frame / predictions
latest_frame: Union[VideoFrame, None] = None
latest_predictions: Dict[str, Any] = None

# Get environment variables
VIDEO_REFERENCE = os.getenv('VIDEO_REFERENCE')

# TODO: Check this below
def get_video_reference(reference):
    if reference.isnumeric():
        return int(reference)
    elif reference == "LucidFrameProducer":
        from LucidFrameProducer import LucidFrameProducer
        return LucidFrameProducer
    elif reference == "BaslerFrameProducer":
        from BaslerFrameProducer import BaslerFrameProducer
        return BaslerFrameProducer
    else:
        raise ValueError(f"Unknown VIDEO_REFERENCE: {reference}")

# Usage
VIDEO_REFERENCE = get_video_reference(VIDEO_REFERENCE)
    

# Workflow Params
WORKFLOW_ID = os.getenv('WORKFLOW_ID')
if not WORKFLOW_ID:
    raise ValueError('WORKFLOW_ID environment variable not set')
WORKSPACE_NAME = os.getenv('WORKSPACE_NAME')
if not WORKSPACE_NAME:
    raise ValueError('WORKSPACE_NAME environment variable not set')

def annotate_image(image):
    for pred in latest_predictions[0]:  # Extract the list of predictions from the outer list
        center_x = pred['x']
        center_y = pred['y']
        width = pred['width']
        height = pred['height']
        class_label = pred['class']

        # Calculate the top-left corner of the bounding box
        top_left_x = int(center_x - width / 2)
        top_left_y = int(center_y - height / 2)
        
        # Calculate the bottom-right corner of the bounding box
        bottom_right_x = int(center_x + width / 2)
        bottom_right_y = int(center_y + height / 2)

        # Draw the bounding box
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

        # Put the label above the bounding box
        label_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_x = top_left_x
        label_y = top_left_y - 10 if top_left_y - 10 > 10 else top_left_y + 10  # Ensure label is within image bounds

        # # Draw the label background
        # cv2.rectangle(image, (label_x, label_y - label_size[1]), (label_x + label_size[0], label_y), (0, 255, 0), cv2.FILLED)

        # Put the label text on the image
        cv2.putText(image, class_label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return image

def numpy_array_to_jpeg(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array representing an image")

    # Convert OpenCV image (BGR) to PIL image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Save PIL image as JPEG into a buffer
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=100)
    
    # Get the byte data
    jpeg_bytes = buffered.getvalue()
    
    return jpeg_bytes

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/latest_frame")
def get_latest_frame():
    global latest_frame
    if latest_frame is None:
        return
    _, buffer = cv2.imencode('.jpg', annotate_image(latest_frame.image.copy()), [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    frame_data_uri = f"data:image/jpeg;base64,{frame_base64}"
    return {
        'frame_id': latest_frame.frame_id,
        'data': frame_data_uri
    }

@app.get("/latest_frame_id")
def get_latest_frame_id():
    global latest_frame
    if latest_frame is None:
        return 0
    return latest_frame.frame_id

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Camera Live View</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 20px;
                    background-color: #f8f9fa;
                }
                .container {
                    text-align: center;
                    border: 2px solid #ccc;
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #fff;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }
                h1 {
                    color: #333;
                    margin-bottom: 20px;
                }
                .frame {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                #latest_frame {
                    width: 100%;
                    height: auto;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                .buttons {
                    margin-top: 10px;
                }
                .buttons button {
                    background-color: #8220E5;
                    border: none;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 5px;
                }
                .buttons button:hover {
                    background-color: #c3b6d6;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Camera Live View</h1>
                <div class="frame">
                    <img src="/latest_frame" id="latest_frame">
                    <div class="buttons">
                        <button onclick="restart()">Restart</button>
                        <button onclick="upload()">Upload</button>
                    </div>
                </div>
            </div>

            <script type="text/javascript">
                let latest_frame_id = 0;

                async function fetchLatestFrame() {
                    const latest_frame = await fetch('/latest_frame');
                    const latest_frame_json = await latest_frame.json();
                    document.getElementById('latest_frame').src = latest_frame_json.data;
                    latest_frame_id = latest_frame_json.frame_id;
                }

                async function updateFrame() {
                    const latest_frame_id_response = await fetch('/latest_frame_id');
                    const latest_frame_id_data = await latest_frame_id_response.json();
                    if (latest_frame_id_data > latest_frame_id) {
                        fetchLatestFrame();
                    }
                }

                async function restart() {
                    await fetch('/restart', { method: 'POST' });
                }

                async function upload() {
                    const response = await fetch('/upload', { method: 'POST' });
                    const data = await response.json();
                }

                setInterval(updateFrame, 500);   
                fetchLatestFrame().catch(console.error);
            </script>
        </body>
    </html>
    """

@app.post("/restart")
async def restart():
    client = docker.from_env()
    container_name = "rf_docker"
    try:
        container = client.containers.get(container_name)
        container.restart()
        print(f"Container {container_name} restarted successfully.")
    except docker.errors.NotFound:
        print(f"Container {container_name} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

@app.post("/upload")
async def upload():
    API_KEY = os.getenv('API_KEY')
    split = "train"
    image_name = "image_upload"
    if API_KEY is None:
        return {"status": "failed", "detail": "API key not found"}

    url_list = [
        f"https://api.roboflow.com/dataset/{WORKFLOW_ID}/upload",
        "?api_key=" + API_KEY,
        f"&batch=Uploaded Via Roboflow Edge"
    ]

    # Construct the URL
    url = "".join(url_list)

    try:
        # Convert the image to base64
        img_str = numpy_array_to_jpeg(latest_frame.image)

        # POST to the API
        m = MultipartEncoder(
            fields={
                "name": image_name,
                "split": split,
                "file": ("imageToUpload", img_str, "image/jpeg"),
            }
        )

        response = requests.post(url, data=m, headers={"Content-Type": m.content_type}, timeout=(300, 300))

        return response.json()

    except Exception as e:
        return {"status": "failed", "detail": str(e)}

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start the server thread
server_thread = threading.Thread(target=run_server)
server_thread.daemon = True
server_thread.start()

# Callback for predictions
def on_prediction(predictions, frame: VideoFrame):
    global latest_frame, latest_predictions
    latest_frame = frame
    latest_predictions = predictions['predictions']
    print(frame.frame_id, predictions)

# Run inference pipeline
pipeline = InferencePipeline.init_with_workflow(
    workspace_name=WORKSPACE_NAME,
    workflow_id=WORKFLOW_ID,
    video_reference=VIDEO_REFERENCE,
    on_prediction=on_prediction,
    source_buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
    source_buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
)

pipeline.start()
pipeline.join()
os._exit(0)