import torch

try:
    GPU_DEVICE_NAME = torch.cuda.get_device_name()
    GPU_DEVICE_COUNT = torch.cuda.device_count()
except Exception:
    GPU_DEVICE_NAME = "cpu"
    GPU_DEVICE_COUNT = 0

def set_info(self):
    response = get_model_name(self)
    miner_info = {
        "model_name": response,
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
    return "dummy model"

def get_mode(self):
    if self.config.researcher:
        return "researcher"
    return "regular"

