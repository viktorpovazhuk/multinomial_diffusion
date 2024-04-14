import os
import errno
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms
from PIL import Image
import os

from pathlib import Path

import pandas as pd
import cv2

from albumentations import HorizontalFlip


ROOT = os.path.dirname(os.path.abspath(__file__))


class EDA20KFast(data.Dataset):
    def __init__(self, root=ROOT, split='train', resolution=(128, 128)):
        self.resolution = resolution
        self.root = os.path.expanduser(root)
        self.images_dir = Path(self.root) / 'images' / split
        self.transform = HorizontalFlip()
        self.split = split
        metadata = pd.read_csv(Path(self.root) / 'metadata.csv')
        self.metadata = metadata[metadata['split'] == split]

        if split not in ('train', 'val', 'test'):
            raise ValueError('split should be one of {train, val, test}')

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        
        img = cv2.imread(str(self.images_dir / f'{row["img_stem"]}.png'), flags=cv2.IMREAD_UNCHANGED)

        if img.shape != self.resolution:
            img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_NEAREST)

        if self.transform:
            img = self.transform(image=img)['image']

        img = torch.tensor(img).long()

        img = img.unsqueeze(0)

        return img

    def __len__(self):
        return len(self.metadata)