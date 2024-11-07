"""Urban Sounds 8k dataset from https://urbansounddataset.weebly.com/urbansound8k.html."""

from bs4 import BeautifulSoup
import numpy as np
import os
import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class UrbanSound8k(Dataset):
    """Urban Sound 8k dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Download the data to the root dir if it does not exist
        def download(filename, source_url, file_id):
            print(f"Downloading {source_url} to {filename}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            download_file_from_google_drive(file_id, filename)

        videos_file_name = os.path.join(root_dir, "UrbanSound8k/urbansound8k.npz")
        _AUDIO_URL = "https://drive.google.com/uc?export=view&id=186mCHedZ_hfnTazOCFsPFUshxGFzd3qV"
        _AUDIO_FILE_ID = "186mCHedZ_hfnTazOCFsPFUshxGFzd3qV"

        if not os.path.isfile(videos_file_name):
            download(videos_file_name, _AUDIO_URL, _AUDIO_FILE_ID)

        with np.load(videos_file_name, allow_pickle=True) as npz:
            audio_np = npz["mel_spectrograms"]
            labels_np = npz["labels"]

        self._audio_data = torch.from_numpy(audio_np)
        self._labels_data = torch.from_numpy(labels_np)

    def __len__(self):
        return self._audio_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio = self._audio_data[idx]
        label = self._labels_data[idx]
        if self.transform:
            audio = self.transform(audio)
        return audio, label


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        content_length = int(response.headers["Content-Length"])
        total_chunks = (
            (content_length // CHUNK_SIZE) + 1
            if content_length % CHUNK_SIZE != 0
            else 0
        )
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE), total=total_chunks):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id, "confirm": 1}, stream=True)
    token = get_confirm_token(response)

    content_type = response.headers["Content-Type"]
    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    else:
        if content_type.startswith("text/html"):
            # Second download for large file virus warning
            html_content = response.text
            assert html_content.startswith(
                "<!DOCTYPE html><html><head><title>Google Drive - Virus scan warning"
            )

            soup = BeautifulSoup(html_content, features="html.parser")
            form_tag = soup.find("form", {"id": "download-form"})
            download_url = form_tag["action"]

            # Get all of the attributes
            id = soup.find("input", {"name": "id"})["value"]
            export = soup.find("input", {"name": "export"})["value"]
            confirm = soup.find("input", {"name": "confirm"})["value"]
            uuid = soup.find("input", {"name": "uuid"})["value"]
            params = {
                "id": id,
                "export": export,
                "confirm": confirm,
                "uuid": uuid,
            }
            response = session.get(download_url, params=params, stream=True)
    save_response_content(response, destination)
