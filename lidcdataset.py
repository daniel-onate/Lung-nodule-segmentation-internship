import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LIDCDataset(Dataset):

    def __init__(self, img_dir, mask_dir):

        self.image_dir = img_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        mask = mask > 0
        mask = mask.astype(np.float32)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask