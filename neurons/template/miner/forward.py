import torch

try:
    GPU_DEVICE_NAME = torch.cuda.get_device_name()
    GPU_DEVICE_COUNT = torch.cuda.device_count()
except Exception:
    GPU_DEVICE_NAME = "cpu"
    GPU_DEVICE_COUNT = 0

def set_info(self):
    # Set information of miner
    # Currently only model name is set
    response = get_model_name(self)
    miner_info = {
        "model_name": response,
        "min_stake": 100,
        "volume_per_validator": 100,
        "device_info": {
            "gpu_device_name": GPU_DEVICE_NAME,
            "gpu_device_count": GPU_DEVICE_COUNT,
        }
    }
    return miner_info


def get_model_name(self):
    # Logic to handle querying models name using ML model watermark (?)
    return "dummy model"

