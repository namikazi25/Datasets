import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MochegDataset(Dataset):
    """
    Custom dataset for loading image-text pairs from MOCHEG data.
    """

    def __init__(self, images_dir, img_evidence_csv, corpus_csv, transform=None, limit=None):
        """
        Args:
            images_dir (str): Directory where images are stored.
            img_evidence_csv (str): Path to img_evidence_qrels.csv.
            corpus_csv (str): Path to Corpus2.csv.
            transform (callable, optional): Transformations for images.
            limit (int, optional): Number of samples to load (for testing purposes).
        """
        self.images_dir = images_dir
        self.img_evidence_data = pd.read_csv(img_evidence_csv)
        self.corpus_data = pd.read_csv(corpus_csv)
        self.transform = transform

        # Merge data based on TOPIC/claim_id
        self.data = self._prepare_data()

        # Apply the limit if specified
        if limit is not None:
            self.data = self.data.head(limit)

    def _prepare_data(self):
        """
        Merge img_evidence_qrels with Corpus2 based on the TOPIC/claim_id.
        """
        merged_data = self.img_evidence_data.merge(
            self.corpus_data, left_on="TOPIC", right_on="claim_id", how="inner"
        )
        return merged_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image path and headline
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_dir, row["DOCUMENT#"])
        headline = row["Headline"]

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, headline

    def check_integrity(self):
        """
        Check if all images and text references are valid.
        """
        missing_images = []
        for idx, row in self.data.iterrows():
            image_path = os.path.join(self.images_dir, row["DOCUMENT#"])
            if not os.path.exists(image_path):
                missing_images.append(image_path)

        if missing_images:
            print("Missing Images:")
            for path in missing_images:
                print(f"- {path}")
            return False
        else:
            print("All images are present.")
            return True
