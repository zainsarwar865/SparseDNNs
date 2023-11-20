import os
import pandas as pd
from torchvision.io import read_image
#from skimage.io import imread
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if image.shape[0] == 4:
            image = image[:3, :, :]
        label = self.df.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    


class CustomImageDataset_Adv(Dataset):
    def __init__(self, dataset_tuple, transform=None, target_transform=None):
        self.data = dataset_tuple[0]
        self.labels = dataset_tuple[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label
    



