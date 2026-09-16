"""Fixed corruption stress test; does not tune on test labels or adapt weights."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screening.data import ROOT, SegmentationDataset
from screening.model import build_model
from screening.run import evaluate

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); a = p.parse_args()
    run = ROOT/a.run
    torch.set_num_threads(4)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    ck = torch.load(run/'best.pt', map_location=device, weights_only=False)
    model = build_model().to(device); model.load_state_dict(ck['model'])
    result = {}
    for condition in ['clean', 'brightness_0.5', 'blur_sigma2', 'noise_sd20']:
        ds = SegmentationDataset(ROOT/ck['config']['manifest'], 'test', ck['config']['size'])
        rng = np.random.default_rng(2026)
        if condition == 'brightness_0.5': ds.images = [(x*.5).astype('uint8') for x in ds.images]
        if condition == 'blur_sigma2': ds.images = [cv2.GaussianBlur(x,(0,0),2) for x in ds.images]
        if condition == 'noise_sd20': ds.images = [np.clip(x.astype(float)+rng.normal(0,20,x.shape),0,255).astype('uint8') for x in ds.images]
        result[condition] = evaluate(model, ds, device, ck['config']['batch_size'])
    (run/'robustness.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))
