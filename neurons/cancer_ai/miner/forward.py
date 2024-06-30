import torch
import requests
import bittensor as bt

from io import BytesIO
from PIL import Image

try:
    GPU_DEVICE_NAME = torch.cuda.get_device_name()
    GPU_DEVICE_COUNT = torch.cuda.device_count()
except Exception:
    GPU_DEVICE_NAME = "cpu"
    GPU_DEVICE_COUNT = 0

def set_info(self):
    # response = get_model_name(self)
    miner_info = {
        # "model_name": response,
        "min_stake": self.config.min_stake,
        "device_info": {
            "gpu_device_name": GPU_DEVICE_NAME,
            "gpu_device_count": GPU_DEVICE_COUNT,
        },
        "miner_mode": get_mode(self)
    }
    return miner_info

def get_model_name(self):
    #TODO: Logic to handle querying models name using ML model watermark (?)
    ...

def get_mode(self):
    if self.config.researcher:
        return "researcher"
    return "regular"

def get_images(images: list):
    images = []
    
    for image in images:
        try:
            response = requests.get(image["image_url"])
            response.raise_for_status()
            if 'image/jpeg' in response.headers.get('Content-Type', ''):
                jpg_image = response.content
                images.append(image["id"], jpg_image)
            else:
                bt.logging.error(f"URL does not point to a JPEG image: {image["image_url"]}")
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Error fetching {image["image_url"]}: {e}")
        except IOError as e:
            bt.logging.error(f"Error opening image from {image["image_url"]}: {e}")

    return images

def get_image(self, image: str):
    try:
        response = requests.get(self.config.dataset_api + image)
        response.raise_for_status()
        if 'image/jpeg' in response.headers.get('Content-Type', ''):
            jpg_image = response.content
        else:
            bt.logging.error(f"URL does not point to a JPEG image: {image}")

    except requests.exceptions.RequestException as e:
        bt.logging.error(f"Error fetching image from {image}: {e}")
    except IOError as e:
        bt.logging.error(f"Error opening image from {image}: {e}")
    
    return jpg_image