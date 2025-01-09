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
import urllib.request

from xdiffusion.datasets.mnist import convert_labels_to_prompts


def load_mnist(
    training_height: int,
    training_width: int,
    split: str = "train",
    invert: bool = False,
) -> Tuple[Dataset, Callable[[torch.Tensor], List[str]]]:
    if split != "train":
        print(
            f"WARNING: This dataset only support a train split. The '{split}' will be the same as the 'train' split."
        )

    if invert:
        xforms = [
            v2.Resize(
                size=(training_height, training_width),
                antialias=True,
            ),
            # Invert the dataset for LoRA training
            v2.Lambda(_invert),
        ]
    else:
        xforms = [
            v2.Resize(
                size=(training_height, training_width),
                antialias=True,
            ),
            v2.ToDtype(torch.float32, scale=True),
        ]
    dataset = MNISTEmbedded(
        ".",
        train=True,
        transform=v2.Compose(xforms),
    )

    return dataset, convert_labels_to_prompts


class MNISTEmbedded(Dataset):

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

        self.local_image_data_fmt = (
            "MNISTEmbeddedGemma2/mnist_embedded_gemma_2_image_data_{shard_idx:03d}.npy"
        )
        self.local_class_labels_fmt = "MNISTEmbeddedGemma2/mnist_embedded_gemma_2_class_labels_{shard_idx:03d}.npy"
        self.local_caption_embeddings_fmt = "MNISTEmbeddedGemma2/mnist_embedded_gemma_2_caption_embeddings_{shard_idx:03d}.npy"
        self.local_caption_embedding_attention_masks_fmt = "MNISTEmbeddedGemma2/mnist_embedded_gemma_2_caption_embedding_attention_masks_{shard_idx:03d}.npy"
        self.num_shards = 24

        os.makedirs(os.path.join(self.root_dir, "MNISTEmbeddedGemma2"), exist_ok=True)

        shard_urls_and_ids = [
            {
                "image_data_url": f"https://s3.us-west-2.amazonaws.com/xdiffusion.datasets/mnist_embedded_gemma_2/mnist_embedded_gemma_2_image_data_{shard_idx:03d}.npy",
                "class_labels_url": f"https://s3.us-west-2.amazonaws.com/xdiffusion.datasets/mnist_embedded_gemma_2/mnist_embedded_gemma_2_class_labels_{shard_idx:03d}.npy",
                "caption_embeddings_url": f"https://s3.us-west-2.amazonaws.com/xdiffusion.datasets/mnist_embedded_gemma_2/mnist_embedded_gemma_2_caption_embeddings_{shard_idx:03d}.npy",
                "caption_embedding_attention_masks_url": f"https://s3.us-west-2.amazonaws.com/xdiffusion.datasets/mnist_embedded_gemma_2/mnist_embedded_gemma_2_caption_embedding_attention_masks_{shard_idx:03d}.npy",
            }
            for shard_idx in range(self.num_shards)
        ]

        self.examples_per_shard = -1
        self.total_examples = -1

        # Download the data to the root dir if it does not exist
        for shard_idx in range(self.num_shards):
            image_data_file_name = os.path.join(
                root_dir, self.local_image_data_fmt.format(shard_idx=shard_idx)
            )
            image_data_url = shard_urls_and_ids[shard_idx]["image_data_url"]

            labels_file_name = os.path.join(
                root_dir, self.local_class_labels_fmt.format(shard_idx=shard_idx)
            )
            class_labels_url = shard_urls_and_ids[shard_idx]["class_labels_url"]

            caption_embeddings_file_name = os.path.join(
                root_dir, self.local_caption_embeddings_fmt.format(shard_idx=shard_idx)
            )
            caption_embeddings_url = shard_urls_and_ids[shard_idx][
                "caption_embeddings_url"
            ]

            caption_embedding_attention_masks_file_name = os.path.join(
                root_dir,
                self.local_caption_embedding_attention_masks_fmt.format(
                    shard_idx=shard_idx
                ),
            )
            caption_embedding_attention_masks_url = shard_urls_and_ids[shard_idx][
                "caption_embedding_attention_masks_url"
            ]

            if not os.path.isfile(image_data_file_name):
                download_from_http(image_data_url, image_data_file_name)
            if not os.path.isfile(labels_file_name):
                download_from_http(class_labels_url, labels_file_name)
            if not os.path.isfile(caption_embeddings_file_name):
                download_from_http(
                    caption_embeddings_url,
                    caption_embeddings_file_name,
                )
            if not os.path.isfile(caption_embedding_attention_masks_file_name):
                download_from_http(
                    caption_embedding_attention_masks_url,
                    caption_embedding_attention_masks_file_name,
                )

            # Load the files and count the number of examples in each shard
            # file
            image_data = np.load(
                image_data_file_name,
                mmap_mode="r",
            )
            class_labels = np.load(
                labels_file_name,
                mmap_mode="r",
            )
            caption_embeddings = np.load(
                caption_embeddings_file_name,
                mmap_mode="r",
            )
            caption_embedding_attention_masks = np.load(
                caption_embedding_attention_masks_file_name,
                mmap_mode="r",
            )

            examples_per_shard = image_data.shape[0]
            assert class_labels.shape[0] == examples_per_shard
            assert caption_embeddings.shape[0] == examples_per_shard
            assert caption_embedding_attention_masks.shape[0] == examples_per_shard

            if shard_idx == 0:
                # This is the first shard
                self.examples_per_shard = examples_per_shard
                self.total_examples = examples_per_shard
            elif shard_idx == (self.num_shards - 1):
                # This is the last shard
                self.total_examples += examples_per_shard
            else:
                assert examples_per_shard == self.examples_per_shard
                self.total_examples += examples_per_shard

            image_data._mmap.close()
            class_labels._mmap.close()
            caption_embeddings._mmap.close()
            caption_embedding_attention_masks._mmap.close()

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Find out which shard this belongs to
        shard_idx = idx // self.examples_per_shard

        # Find out the index inside the shard this belongs to
        example_idx = idx - (shard_idx * self.examples_per_shard)

        image_data_file_name = os.path.join(
            self.root_dir, self.local_image_data_fmt.format(shard_idx=shard_idx)
        )
        class_labels_file_name = os.path.join(
            self.root_dir, self.local_class_labels_fmt.format(shard_idx=shard_idx)
        )
        caption_embeddings_file_name = os.path.join(
            self.root_dir, self.local_caption_embeddings_fmt.format(shard_idx=shard_idx)
        )
        caption_embedding_attention_masks_file_name = os.path.join(
            self.root_dir,
            self.local_caption_embedding_attention_masks_fmt.format(
                shard_idx=shard_idx
            ),
        )

        image_data = np.load(
            image_data_file_name,
            mmap_mode="r",
        )
        label_data = np.load(
            class_labels_file_name,
            mmap_mode="r",
        )
        embedding_data = np.load(
            caption_embeddings_file_name,
            mmap_mode="r",
        )
        mask_data = np.load(
            caption_embedding_attention_masks_file_name,
            mmap_mode="r",
        )

        rv = (
            torch.from_numpy(image_data[example_idx].copy()),
            torch.tensor(label_data[example_idx]),
            {
                "text_embeddings": torch.from_numpy(embedding_data[example_idx].copy()).squeeze(0),
                "text_attention_mask": torch.from_numpy(mask_data[example_idx].copy()).squeeze(0),
            },
        )

        image_data._mmap.close()
        label_data._mmap.close()
        embedding_data._mmap.close()
        mask_data._mmap.close()

        return rv


def download_from_http(url, filename):
    with tqdm(total=1, unit="B", unit_scale=True, desc="Downloading") as pbar:

        def reporthook(blocknum, blocksize, totalsize):
            # Calculate progress
            pbar.total = totalsize
            pbar.refresh()
            progress = blocknum * blocksize
            if totalsize > 0:
                percent = min(int(progress * 100 / totalsize), 100)
                pbar.update(progress - pbar.n)

        urllib.request.urlretrieve(url, filename, reporthook=reporthook)


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


def _invert(x: torch.Tensor) -> torch.Tensor:
    return v2.functional.invert(x)
