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
import supervision as sv
import time
# Import Camera Frame Grabbers
from LucidFrameProducer import LucidFrameProducer
from BaslerFrameProducer import BaslerFrameProducer
import logging

import json
from typing import List, Dict, Any

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
##### FIX
VIDEO_REFERENCE = os.getenv('VIDEO_REFERENCE')
# VIDEO_REFERENCE = "/Users/reed/Documents/VSCode/RoboflowDemos/csx_video.mp4" #"0"

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
        return reference
        ###### FIX - uncomment
        raise ValueError(f"Unknown VIDEO_REFERENCE: {reference}")

# Usage
VIDEO_REFERENCE = get_video_reference(VIDEO_REFERENCE)

# if not VIDEO_REFERENCE:
#     raise ValueError('VIDEO_REFERENCE environment variable not set')
# if VIDEO_REFERENCE.isnumeric():
#     VIDEO_REFERENCE = int(VIDEO_REFERENCE)
# elif VIDEO_REFERENCE == "LucidFrameProducer":
#     from LucidFrameProducer import LucidFrameProducer
#     VIDEO_REFERENCE = LucidFrameProducer
# elif VIDEO_REFERENCE == "BaslerFrameProducer":
#     from BaslerFrameProducer import BaslerFrameProducer
#     VIDEO_REFERENCE = BaslerFrameProducer
    

# Workflow Params
##### FIX - uncomment
WORKFLOW_ID = os.getenv('WORKFLOW_ID')
if not WORKFLOW_ID:
    raise ValueError('WORKFLOW_ID environment variable not set')
WORKSPACE_NAME = os.getenv('WORKSPACE_NAME')
if not WORKSPACE_NAME:
    raise ValueError('WORKSPACE_NAME environment variable not set')
# WORKFLOW_ID = "chain-models"
# WORKSPACE_NAME = "intermodal"


###### Printing Info ######
# def combine_ocr_characters(predictions: List[Dict[str, Any]]) -> str:
#     """Combines OCR characters into a single string based on their positions and class names.

#     Args:
#         predictions (List[Dict[str, Any]]): List of OCR character predictions.

#     Returns:
#         str: Combined OCR string.
#     """
#     sorted_predictions = sorted(predictions, key=lambda p: (p['y'], p['x']))
#     return ''.join([pred['class'] for pred in sorted_predictions])

# def process_json_response(response: List[Dict[str, Any]]) -> None:
#     """Processes the JSON response and prints the OCR strings in a pretty format.

#     Args:
#         response (List[Dict[str, Any]]): JSON response from the CV model.
#     """
#     print(response)
#     # for item in response:
#     print("================================")
#     print(response)
#     image_info = response.get('predictions', {}).get('image', {})
#     print("--------------------")
#     image_width = image_info.get('width')
#     image_height = image_info.get('height')
#     if image_width is not None and image_height is not None:
#         print("Image size:", image_width, "x", image_height)
#     else:
#         print("Image size: Not available")
    
#     for crop in item.get('crops_predictions', []):
#         parent_id = crop['predictions'][0].get('parent_id') if crop['predictions'] else None
#         if parent_id:
#             parent_prediction = next((pred for pred in item['predictions']['predictions'] if pred['detection_id'] == parent_id), None)
#             if parent_prediction:
#                 if 'vertical' in parent_prediction['class']:
#                     sorted_predictions = sorted(crop['predictions'], key=lambda p: (p['y'], p['x']))
#                 else:
#                     sorted_predictions = sorted(crop['predictions'], key=lambda p: (p['x'], p['y']))
                
#                 combined_string = ''.join([pred['class'] for pred in sorted_predictions])
#                 print(f"Detected String ({parent_prediction['class']}): {combined_string}")
#             else:
#                 print("Parent prediction not found for parent_id:", parent_id)
#         else:
#             print("No valid predictions found in crop")

class Detections:
    def __init__(self, xyxy: np.ndarray, mask: Any, confidence: np.ndarray, class_id: np.ndarray, tracker_id: Any, data: Dict[str, Any]):
        self.xyxy = xyxy
        self.mask = mask
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id
        self.data = data

def combine_ocr_characters(predictions: Detections) -> str:
    """Combines OCR characters into a single string based on their positions and class names.

    Args:
        predictions (Detections): OCR character predictions.

    Returns:
        str: Combined OCR string.
    """
    sorted_indices = np.argsort(predictions.xyxy[:, 1])  # Sort by y-coordinate (top to bottom)
    sorted_predictions = [predictions.data['class_name'][i] for i in sorted_indices]
    return ''.join(sorted_predictions)

def process_json_response(response: List[Dict[str, Any]]) -> None:
    """Processes the JSON response and prints the OCR strings in a pretty format.

    Args:
        response (List[Dict[str, Any]]): JSON response from the CV model.
    """
    for item in response:
        image_info = item.get('predictions', {}).data.get('image_dimensions', None)
        if image_info is not None and image_info.size > 0:
            image_width = image_info[0][1]  # assuming it's a list of [height, width]
            image_height = image_info[0][0]
            # print("Image size:", image_width, "x", image_height)
        else:
            print("Image size: Not available")
        
        crops_predictions = item.get('crops_predictions', [])
        detection_offset = item.get('detection_offset', [])
        detection_parent_ids = detection_offset.data.get('parent_id', [])
        detection_class_names = detection_offset.data.get('class_name', [])
        print('**'*20)
        print("OCR Detections:")
        for cnt_i, crop in enumerate(crops_predictions):
            parent_ids = crop.data.get('parent_id', [])
            if parent_ids.size == 0:
                print("No parent IDs found for crop.")
                continue
            for parent_id in detection_parent_ids: #parent_ids:
                parent_predictions = item['predictions']
                parent_data = parent_predictions.data
                parent_idx_array = np.where(parent_data['detection_id'] == parent_id)[0]
                if len(parent_idx_array) == 0:
                    print(f"Warning: Parent ID {parent_id} not found in parent data.")
                    continue
                parent_idx = parent_idx_array[0]
                parent_class = detection_class_names[cnt_i] #parent_data['class_name'][parent_idx]

                if 'vertical' in parent_class:
                    sorted_indices = np.argsort(crop.xyxy[:, 1])  # Sort by y-coordinate (top to bottom)
                else:
                    sorted_indices = np.argsort(crop.xyxy[:, 0])  # Sort by x-coordinate (left to right)
                
                sorted_predictions = [crop.data['class_name'][i] for i in sorted_indices]
                combined_string = ''.join(sorted_predictions)
                print(f"\t{parent_class}: {combined_string}")
                # Need to only do once...
                break

        print('**'*20)
###############

def annotate_image(image):
    # print(latest_predictions)
    # detections = sv.Detections.from_inference(latest_predictions)
    detections = latest_predictions

    annotated_image = sv.BoundingBoxAnnotator().annotate(
    scene=image.copy(), detections=detections
    )
    annotated_image = sv.LabelAnnotator().annotate(
        scene=annotated_image, detections=detections
    )
    image = annotated_image
    # sv.plot_image(annotated_image)
    # for pred in latest_predictions[0]:  # Extract the list of predictions from the outer list
    #     center_x = pred['x']
    #     center_y = pred['y']
    #     width = pred['width']
    #     height = pred['height']
    #     class_label = pred['class']

    #     # Calculate the top-left corner of the bounding box
    #     top_left_x = int(center_x - width / 2)
    #     top_left_y = int(center_y - height / 2)
        
    #     # Calculate the bottom-right corner of the bounding box
    #     bottom_right_x = int(center_x + width / 2)
    #     bottom_right_y = int(center_y + height / 2)

    #     # Draw the bounding box
    #     cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    #     # Put the label above the bounding box
    #     label_size = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    #     label_x = top_left_x
    #     label_y = top_left_y - 10 if top_left_y - 10 > 10 else top_left_y + 10  # Ensure label is within image bounds

    #     # # Draw the label background
    #     # cv2.rectangle(image, (label_x, label_y - label_size[1]), (label_x + label_size[0], label_y), (0, 255, 0), cv2.FILLED)

    #     # Put the label text on the image
    #     cv2.putText(image, class_label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

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
    print(f"Processed New Image: {frame.frame_id}")
    # print(frame.frame_id, predictions)
    # print("Processed JSON Response: ")
    process_json_response([predictions])

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