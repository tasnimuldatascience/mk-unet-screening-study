import numpy as np
import torch
import torch.nn.functional as F

def structure_loss(logits, mask):
    weights = 1 + 5 * (F.avg_pool2d(mask, 31, 1, 15) - mask).abs()
    bce = (weights * F.binary_cross_entropy_with_logits(logits, mask, reduction='none')).sum((2, 3)) / weights.sum((2, 3))
    pred = logits.sigmoid()
    intersection = (weights * pred * mask).sum((2, 3))
    union = (weights * (pred + mask)).sum((2, 3))
    return (bce + 1 - (intersection + 1) / (union - intersection + 1)).mean()

def binary_metrics(pred, target):
    p, t = np.asarray(pred, bool), np.asarray(target, bool)
    if p.shape != t.shape:
        raise ValueError('Prediction and target shapes differ')
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    return dict(dice=(2*tp+1e-6)/(2*tp+fp+fn+1e-6), iou=(tp+1e-6)/(tp+fp+fn+1e-6), tp=tp, fp=fp, fn=fn)

def summarize(rows):
    if not rows:
        raise ValueError('No evaluation samples')
    values = np.array([r['dice'] for r in rows])
    rng = np.random.default_rng(42)
    ci = np.quantile(rng.choice(values, (2000, len(values)), replace=True).mean(1), [.025, .975])
    return dict(n=len(rows), dice=float(values.mean()), iou=float(np.mean([r['iou'] for r in rows])),
                dice_bootstrap95=ci.tolist(), uncertainty_unit='image; correlated frames may make CI optimistic')
