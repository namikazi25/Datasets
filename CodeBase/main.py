import os
from torch.utils.data import DataLoader
from dataset.mocheg_dataset import MochegDataset
from models.instructblip_model import InstructBLIP

import torch
print(torch.__version__)
print(torch.cuda.is_available())

def main():
    # Paths
    #images_dir = "/content/drive/MyDrive/MOCHEG/extracted/mocheg/train/images"  # Update with your images folder path
    images_dir = "/content/drive/MyDrive/MOCHEG/extracted/mocheg/images"
    corpus_csv = "/content/drive/MyDrive/MOCHEG/extracted/mocheg/train/Corpus2.csv"  # Update with your text CSV path
    img_evidence_csv = "/content/drive/MyDrive/MOCHEG/extracted/mocheg/train/img_evidence_qrels.csv"

    # Initialize dataset with a limit of 50 samples
    dataset = MochegDataset(images_dir, img_evidence_csv, corpus_csv, limit=50)

    # Check data integrity
    if not dataset.check_integrity():
        print("Dataset integrity check failed. Please fix the dataset before proceeding.")
        return

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Example usage of dataloader
    for batch in dataloader:
        images, texts = batch
        print(f"Images: {len(images)}, Texts: {texts}")
        break

if __name__ == "__main__":
    main()
