"""Moving MNIST dataset from http://www.cs.toronto.edu/~nitish/unsupervised_video.

Augments the original Moving MNIST dataset with labels for text guided video diffusion.

Based on an implementation of unconditional MovingMNIST generation from:
https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe
"""

from bs4 import BeautifulSoup
import numpy as np
import os
import requests
import sys
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm
from typing import Callable, List, Tuple


def load_moving_mnist_image(
    training_height: int,
    training_width: int,
    split: str = "train",
    invert: bool = False,
) -> Tuple[Dataset, Callable[[torch.Tensor], List[str]]]:
    assert split in ["train", "validation"]

    if invert:
        xforms = [
            # To the memory requirements, resize the MNIST
            # images from (64,64) to (32, 32).
            v2.Resize(
                size=(training_height, training_width),
                antialias=True,
            ),
            # Convert the motion images to (0,1) float range
            v2.ToDtype(torch.float32, scale=True),
            # Invert the dataset for LoRA training
            v2.Lambda(_invert),
        ]
    else:
        xforms = [  # To the memory requirements, resize the MNIST
            # images from (64,64) to (32, 32).
            v2.Resize(
                size=(training_height, training_width),
                antialias=True,
            ),
            # Convert the motion images to (0,1) float range
            v2.ToDtype(torch.float32, scale=True),
        ]
    if split == "train":
        dataset = MovingMNISTImage(
            ".",
            train=True,
            transform=v2.Compose(xforms),
        )

    else:
        dataset = MovingMNISTImage(
            ".",
            train=False,
            transform=v2.Compose(xforms),
        )
    return dataset, convert_labels_to_prompts


class MovingMNIST(Dataset):
    """Moving MNIST dataset."""

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
        from urllib.request import urlretrieve

        def download(filename, source_url, file_id):
            print(f"Downloading {source_url} to {filename}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            download_file_from_google_drive(file_id, filename)

        videos_file_name = os.path.join(root_dir, "MovingMNIST/videos_data.npz")
        labels_file_name = os.path.join(root_dir, "MovingMNIST/labels_data.npz")

        _VIDEOS_URL = "https://drive.google.com/uc?export=view&id=1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _LABELS_URL = "https://drive.google.com/uc?export=view&id=17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"
        _VIDEOS_FILE_ID = "1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _LABELS_FILE_ID = "17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"

        if not os.path.isfile(videos_file_name):
            download(videos_file_name, _VIDEOS_URL, _VIDEOS_FILE_ID)
        if not os.path.isfile(labels_file_name):
            download(labels_file_name, _LABELS_URL, _LABELS_FILE_ID)

        self._num_frames_per_video = 30
        self._num_videos = 10000
        self._num_digits_per_video = 2

        with np.load(videos_file_name, allow_pickle=True) as npz:
            videos_np = npz[npz.files[0]]

        with np.load(labels_file_name, allow_pickle=True) as npz:
            labels_np = npz[npz.files[0]]

        self._video_data = torch.from_numpy(
            videos_np.reshape(self._num_videos, 1, self._num_frames_per_video, 64, 64)
        )
        self._labels_data = torch.from_numpy(
            labels_np.reshape(
                self._num_videos, self._num_frames_per_video, self._num_digits_per_video
            )[:, 0, :]
        ).squeeze()

    def __len__(self):
        return self._video_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video = self._video_data[idx]
        labels = self._labels_data[idx]
        if self.transform:
            video = self.transform(video)
        return video, labels


class MovingMNISTImage(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, train: bool = True):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Download the data to the root dir if it does not exist
        from urllib.request import urlretrieve

        def download(filename, source_url, file_id):
            print(f"Downloading {source_url} to {filename}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            download_file_from_google_drive(file_id, filename)

        train_videos_file_name = os.path.join(root_dir, "MovingMNIST/videos_data.npz")
        train_labels_file_name = os.path.join(root_dir, "MovingMNIST/labels_data.npz")
        val_videos_file_name = os.path.join(
            root_dir, "MovingMNIST/videos_data_validation.npz"
        )
        val_labels_file_name = os.path.join(
            root_dir, "MovingMNIST/labels_data_validation.npz"
        )

        _TRAIN_VIDEOS_URL = "https://drive.google.com/uc?export=view&id=1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq&confirm=1"
        _TRAIN_LABELS_URL = "https://drive.google.com/uc?export=view&id=17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai&confirm=1"
        _TRAIN_VIDEOS_FILE_ID = "1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _TRAIN_LABELS_FILE_ID = "17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"

        _VAL_VIDEOS_URL = "https://drive.google.com/uc?export=view&id=1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq&confirm=1"
        _VAL_LABELS_URL = "https://drive.google.com/uc?export=view&id=17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai&confirm=1"
        _VAL_VIDEOS_FILE_ID = "1Hii6NpDzAyA4L0wXfOrPr_FTHJGA6RUq"
        _VAL_LABELS_FILE_ID = "17TQWPSiFqPW6I-0nq-LvBy1IY3Jaxmai"

        if train:
            videos_file_name = train_videos_file_name
            labels_file_name = train_labels_file_name

            _VIDEOS_URL = _TRAIN_VIDEOS_URL
            _LABELS_URL = _TRAIN_LABELS_URL
            _VIDEOS_FILE_ID = _TRAIN_VIDEOS_FILE_ID
            _LABELS_FILE_ID = _TRAIN_LABELS_FILE_ID
        else:
            videos_file_name = val_videos_file_name
            labels_file_name = val_labels_file_name
            _VIDEOS_URL = _VAL_VIDEOS_URL
            _LABELS_URL = _VAL_LABELS_URL
            _VIDEOS_FILE_ID = _VAL_VIDEOS_FILE_ID
            _LABELS_FILE_ID = _VAL_LABELS_FILE_ID

        if not os.path.isfile(videos_file_name):
            download(videos_file_name, _VIDEOS_URL, _VIDEOS_FILE_ID)
        if not os.path.isfile(labels_file_name):
            download(labels_file_name, _LABELS_URL, _LABELS_FILE_ID)

        self._num_frames_per_video = 30
        self._num_videos = 10000
        self._num_digits_per_video = 2

        with np.load(videos_file_name) as npz:
            videos_np = npz[npz.files[0]]

        with np.load(labels_file_name) as npz:
            labels_np = npz[npz.files[0]]

        # The video data is (num_videos * num_frames, 1, 64, 64)
        self._video_data = torch.from_numpy(videos_np)
        self._labels_data = torch.from_numpy(labels_np).squeeze()

    def __len__(self):
        return self._video_data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video = self._video_data[idx]
        labels = self._labels_data[idx]
        if self.transform:
            video = self.transform(video)
        return video, labels


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


def convert_labels_to_prompts(labels: torch.Tensor) -> List[str]:
    """Converts MNIST class labels to text prompts.

    Supports both the strings "0" and "zero" to describe the
    class labels.
    """
    # The conditioning we pass to the model will be a vectorized-form of
    # MNIST classes. Since we have a fixed number of classes, we can create
    # a hard-coded "embedding" of the MNIST class label.
    text_labels = [
        ("zero", "0"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
    ]

    # First convert the labels into a list of string prompts
    prompts = [
        f"{text_labels[labels[i][0]][torch.randint(0, len(text_labels[labels[i][0]]), size=())]} and {text_labels[labels[i][1]][torch.randint(0, len(text_labels[labels[i][1]]), size=())]}"
        for i in range(labels.shape[0])
    ]
    return prompts


def _invert(x: torch.Tensor) -> torch.Tensor:
    return v2.functional.invert(x)
