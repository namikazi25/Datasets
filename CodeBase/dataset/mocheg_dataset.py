import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MochegDataset(Dataset):
    """
    Custom Dataset for the Mocheg dataset to pair images with text.

    Args:
        images_dir (str): Path to the directory containing images.
        csv_path (str): Path to the CSV file containing text and image pairs.
        transform (callable, optional): Optional transform to be applied on an image.
    """
    def __init__(self, images_dir, csv_path, transform=None):
        self.images_dir = images_dir
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip()
        print("Columns in CSV:", self.data.columns)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract image and text information
        image_name = self.data.iloc[idx]['evidence_id']
        text = self.data.iloc[idx]['Headline']
        image_path = os.path.join(self.images_dir, image_name)

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, text
