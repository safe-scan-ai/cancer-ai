import requests
from functools import wraps
import bt  # Assuming bt is your logging module or framework

def inject_api_key(api_key_attr):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            if not hasattr(self, api_key_attr):
                raise AttributeError(f"{self.__class__.__name__} object has no attribute '{api_key_attr}'")
            api_key = getattr(self, api_key_attr)
            if 'headers' not in kwargs:
                kwargs['headers'] = {}
            kwargs['headers'].update({"x-api-key": api_key})
            return func(*args, **kwargs)
        return wrapper
    return decorator

class DatasetAPI:
    def __init__(self, base_path, api_key):
        self.base_path = base_path
        self.headers = {"x-api-key": api_key}

    def get_image_data(self, amount, headers):
        """Gets images from API"""
        endpoint = f"{self.base_path}/dataset/skin/melanoma?amount={amount}"
        response = requests.get(url=endpoint, headers=self.headers)
        data = response.json()
        bt.logging.debug(f"Dataset API response: {data}")
        return data

class StatsAPI:
    def __init__(self, base_path, api_key):
        self.base_path = base_path
        self.headers = {"x-api-key": api_key}

    def send_researcher_scores(self, researcher_uid, researcher_scores, headers):

        endpoint = f"{self.base_path}/researcher/testing/{researcher_uid}"
        res = requests.post(url=endpoint, json=researcher_scores, headers=self.headers)
        return res

    def fetch_top_researchers(self, headers):
        endpoint = f"{self.base_path}/emission-share"
        response = requests.get(url=endpoint, headers=self.headers)
        data = response.json()
        return data