import os
import json
from typing import List, Tuple, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class HighResDataset(Dataset):
    def __init__(self, image_dir: str, json_file: str, transform: transforms.Compose = None):
        self.image_dir = image_dir
        self.transform = transform
        
        with open(json_file, 'r') as f:
            self.image_info = json.load(f)

        self.image_ids = list(self.image_info.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id = self.image_ids[index]
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        sample = {
            "image_id": image_id,
            "image": image
        }
        return sample
