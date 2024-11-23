import os
from torch.utils.data import DataLoader
from dataset.mocheg_dataset import MochegDataset
from models.instructblip_model import InstructBLIP

import torch
print(torch.__version__)
print(torch.cuda.is_available())

def main():
    # Paths
    images_dir = "C:/Users/share/Downloads/Datasets/MOCHEG/mocheg_with_tweet_2023_03/mocheg/train/images"  # Update with your images folder path
    csv_path = "C:/Users/share/Downloads/Datasets/MOCHEG/mocheg_with_tweet_2023_03/mocheg/train/Corpus2.csv"  # Update with your text CSV path

    # Initialize dataset and DataLoader
    dataset = MochegDataset(images_dir, csv_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize InstructBLIP
    instruct_blip = InstructBLIP()

    # Iterate through DataLoader
    for batch in dataloader:
        images, texts = batch
        for img, txt in zip(images, texts):
            result = instruct_blip.check_consistency(img, txt)
            print(f"Text: {txt}\nConsistency Result: {result}\n")

if __name__ == "__main__":
    main()
