from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ADE20KCorruption(Dataset):
    def __init__(self, data_root, corruption_root, corruption, severity):
        self.images = sorted((Path(corruption_root) / corruption / str(severity) / "validation").glob("*.jpg"))
        self.labels = Path(data_root) / "annotations" / "validation"
        if not self.images:
            raise FileNotFoundError(f"No images in {corruption_root}/{corruption}/{severity}/validation")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        height, width = pixels.shape[:2]
        padded_height, padded_width = (height + 15) // 16 * 16, (width + 15) // 16 * 16
        top, left = (padded_height - height) // 2, (padded_width - width) // 2
        roi = (slice(top, top + height), slice(left, left + width))
        padded = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
        padded[roi] = pixels
        with Image.open(self.labels / f"{path.stem}.png") as image:
            target = np.array(image, dtype=np.int64) - 1
        target[target == -1] = 255
        return torch.from_numpy(padded).permute(2, 0, 1).contiguous(), torch.from_numpy(target), roi
