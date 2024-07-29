import requests
import bittensor as bt

try:
    import subprocess

    # Use subprocess to run nvidia-smi command and capture its output
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        GPU_DEVICE_NAME = result.stdout.strip()
        GPU_DEVICE_COUNT = len(GPU_DEVICE_NAME.split('\n'))
    else:
        GPU_DEVICE_NAME = "cpu"
        GPU_DEVICE_COUNT = 0
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

def get_images(self, images_urls: dict):
    images = []

    for image_id, image_url in images_urls.items():
        try:
            response = requests.get(self.config.dataset_api + image_url)
            response.raise_for_status()
            if 'image/jpeg' in response.headers.get('Content-Type', ''):
                jpg_image = response.content
                images.append((image_id, jpg_image))
            else:
                bt.logging.error(f"URL does not point to a JPEG image: {image_url}")
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"Error fetching {image_url}: {e}")
        except IOError as e:
            bt.logging.error(f"Error opening image from {image_url}: {e}")

    return images

def get_image(self, image_url: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        if 'image/jpeg' in response.headers.get('Content-Type', ''):
            jpg_image = response.content
        else:
            bt.logging.error(f"URL does not point to a JPEG image: {image_url}")

    except requests.exceptions.RequestException as e:
        bt.logging.error(f"Error fetching image from {image_url}: {e}")
    except IOError as e:
        bt.logging.error(f"Error opening image from {image_url}: {e}")
    
    return jpg_image