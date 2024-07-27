import requests
import bittensor as bt


API_TIMEOUT_S = 5


class DatasetAPI:
    """Class for Dataset API, which
    - fetches synthetic data for miners
    - fetches test data for researchers
    """

    def __init__(self, base_path, api_key):
        self.base_path = base_path
        self.headers = {"x-api-key": api_key}

    def get_image_data(self, amount):
        """Gets images from API"""
        endpoint = f"{self.base_path}/dataset/skin/melanoma?amount={amount}"
        response = requests.get(
            url=endpoint, headers=self.headers, timeout=API_TIMEOUT_S
        )
        data = response.json()
        bt.logging.debug(f"Dataset API response: {data}")
        return data


class StatsAPI:
    """Class for Stats API, which does the following
    - Send testing data from researcher
    - Fetch top researcher for incentive
    """

    def __init__(self, base_path, api_key):
        self.base_path = base_path
        self.headers = {"x-api-key": api_key}

    def send_researcher_scores(self, researcher_uid, researcher_scores):
        """Send testing data from researcher"""
        endpoint = f"{self.base_path}/researcher/testing/{researcher_uid}"
        res = requests.post(
            url=endpoint,
            json=researcher_scores,
            headers=self.headers,
            timeout=API_TIMEOUT_S,
        )
        return res

    def fetch_top_researchers(self):
        """Fetches top researcher for incentive"""
        endpoint = f"{self.base_path}/emission-share"
        response = requests.get(
            url=endpoint, headers=self.headers, timeout=API_TIMEOUT_S
        )
        data = response.json()
        return data
