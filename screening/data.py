import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_rows(rows):
    seen_ids, seen_images = {}, {}
    for r in rows:
        if r['split'] not in ('train', 'val', 'test'):
            raise ValueError('Invalid split')
        if r['id'] in seen_ids:
            raise ValueError(f"Duplicate sample: {r['id']}")
        seen_ids[r['id']] = r['split']
        p, q = ROOT / r['image'], ROOT / r['mask']
        x, y = cv2.imread(str(p)), cv2.imread(str(q), 0)
        if x is None or y is None or x.shape[:2] != y.shape:
            raise ValueError(f"Invalid pair: {r['id']}")
        digest = sha256(p)
        if digest in seen_images and seen_images[digest] != r['split']:
            raise ValueError(f"Image leakage: {r['id']}")
        seen_images[digest] = r['split']
        r['image_sha256'], r['mask_sha256'] = digest, sha256(q)
    if set(r['split'] for r in rows) != {'train', 'val', 'test'}:
        raise ValueError('All three splits must be nonempty')
    return rows

class SegmentationDataset(Dataset):
    def __init__(self, manifest, split, size, augment=False):
        self.rows = [r for r in json.loads(Path(manifest).read_text())['samples'] if r['split'] == split]
        if not self.rows:
            raise ValueError(f'Empty split: {split}')
        self.size, self.augment = size, augment
        self.images, self.masks, self.originals = [], [], []
        for r in self.rows:
            for kind in ('image', 'mask'):
                if r.get(f'{kind}_sha256') and sha256(ROOT/r[kind]) != r[f'{kind}_sha256']:
                    raise ValueError(f"File changed since data preparation: {r[kind]}")
            image = cv2.cvtColor(cv2.imread(str(ROOT / r['image'])), cv2.COLOR_BGR2RGB)
            raw = cv2.imread(str(ROOT / r['mask']), 0)
            mask = (raw > 20).astype(np.float32) if raw.max() > 127 else (raw >= 1).astype(np.float32)
            if r.get('foreground') == 'black':
                mask = 1 - mask
            self.originals.append(mask)
            self.images.append(cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR))
            self.masks.append(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        x, y = self.images[idx], self.masks[idx]
        if self.augment:
            for axis in (0, 1):
                if torch.rand(()).item() < .5:
                    x, y = np.flip(x, axis), np.flip(y, axis)
        x = (x.astype(np.float32) / 255 - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(y[None].copy()), idx
