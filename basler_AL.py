from pypylon import pylon
import cv2
from concurrent.futures import ThreadPoolExecutor
import time
import os
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
from collections import deque
import io
import base64
from PIL import Image
import numpy as np
from datetime import datetime

# Whether to upload frames or not:
upload_bool = True
frame_per_minute = 60 # limit to 60 uploads per minute

# connecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

class RateLimiter:
    def __init__(self, max_rate):
        self.interval = 60.0 / max_rate
        self.last_upload = time.time()

    def __call__(self):
        elapsed = time.time() - self.last_upload
        if elapsed < self.interval:
            return False
        self.last_upload = time.time()
        return True

rate_limiter = RateLimiter(frame_per_minute)  # Rate limit uploads

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

def upload_frame(frame_count, frame):
    """Upload a single frame using API call."""
    if not rate_limiter():
        print("Not yet time for next upload. Skipping this frame.")
        return
    
    API_KEY = os.getenv('API_KEY')
    PROJECT_ID = os.getenv('PROJECT_ID')
    split = "train"
    image_name = f"image_upload_{frame_count}"

    # Get current date and time and format it as a string
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    image_name = f"{timestamp}_frame_{frame_count}"
    
    if API_KEY is None:
        print("API key not found")
        return

    url_list = [
        f"https://api.roboflow.com/dataset/{PROJECT_ID}/upload",
        "?api_key=" + API_KEY,
        f"&batch=Uploaded Via Roboflow Edge"
    ]

    # Construct the URL
    url = "".join(url_list)

    try:
        # Convert the image to base64
        img_str = numpy_array_to_jpeg(frame)

        # POST to the API
        m = MultipartEncoder(
            fields={
                "name": image_name,
                "split": split,
                "file": ("imageToUpload", img_str, "image/jpeg"),
            }
        )

        response = requests.post(url, data=m, headers={"Content-Type": m.content_type}, timeout=(300, 300))

        print(response.json())

    except Exception as e:
        print(f"Upload failed: {str(e)}")

# For multithreading upload:
executor = ThreadPoolExecutor(max_workers=8)  # adjust the number of workers as needed

# Counting images to upload
img_count = 1

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()

        if upload_bool:
            executor.submit(upload_frame, img_count, img)

        # Increase image counter
        img_count += 1
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()
