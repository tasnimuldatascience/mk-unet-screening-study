"""Train, resume and evaluate without using test labels for selection."""
import argparse
import csv
import json
import platform
import random
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data import ROOT, SegmentationDataset, sha256
from .metrics import binary_metrics, structure_loss, summarize
from .model import build_model

def output(model, x):
    y = model(x)
    return y[0] if isinstance(y, list) else y

@torch.inference_mode()
def evaluate(model, dataset, device, batch_size, dest=None):
    model.eval()
    rows, fixed_rows = [], []
    if dest:
        (dest / 'predictions').mkdir(parents=True, exist_ok=True)
    for x, _, ids in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits = output(model, x.to(device))
        for logit, idx in zip(logits, ids.tolist()):
            target = dataset.originals[idx]
            prob = F.interpolate(logit[None], target.shape, mode='bilinear', align_corners=False).sigmoid()[0, 0].cpu().numpy()
            normalized = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
            name = dataset.rows[idx]['id']
            rows.append(dict(id=name, **binary_metrics(normalized >= .5, target)))
            fixed_rows.append(dict(id=name, **binary_metrics(prob >= .5, target)))
            if dest:
                cv2.imwrite(str(dest / 'predictions' / f'{name}.png'), (normalized >= .5).astype('uint8') * 255)
    if dest:
        for name, records in [('per_image.csv', rows), ('per_image_fixed_threshold.csv', fixed_rows)]:
            with (dest / name).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    return dict(normalized=summarize(rows), fixed_threshold=summarize(fixed_rows))

def train(config, resume=False):
    cfg = json.loads(Path(config).read_text())
    run_dir = ROOT / cfg['output']
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'last.pt').exists() and not resume:
        raise ValueError('Run exists; use --resume or a different output directory')
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    model = build_model().to(device)
    manifest = ROOT / cfg['manifest']
    provenance = dict(config=cfg, manifest_sha256=sha256(manifest), python=platform.python_version(),
                      torch=torch.__version__, device=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
                      upstream_commit=subprocess.check_output(['git', '-C', str(ROOT/'vendor/MK-UNet'), 'rev-parse', 'HEAD'], text=True).strip(),
                      model_sha256=sha256(ROOT/'vendor/MK-UNet/mkunet_network.py'), parameters=sum(p.numel() for p in model.parameters()))
    trainset = SegmentationDataset(manifest, 'train', cfg['size'], cfg['augment'])
    valset = SegmentationDataset(manifest, 'val', cfg['size'])
    loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'], eta_min=1e-6) if cfg['scheduler'] == 'cosine' else None
    use_amp = cfg.get('amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    first, best, best_epoch, elapsed = 1, -1., 0, 0.
    if resume:
        ck = torch.load(run_dir / 'last.pt', map_location=device, weights_only=False)
        if ck['config'] != cfg or ck['manifest_sha256'] != provenance['manifest_sha256']:
            raise ValueError('Resume configuration or data manifest differs')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        if sched: sched.load_state_dict(ck['scheduler'])
        first, best, best_epoch, elapsed = ck['epoch']+1, ck['best'], ck['best_epoch'], ck['elapsed']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type == 'cuda': torch.cuda.set_rng_state_all([t.cpu() for t in ck['cuda_rng']])
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng'])
    (run_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    start = time.perf_counter()
    for epoch in range(first, cfg['epochs'] + 1):
        epoch_start = time.perf_counter()
        model.train(); losses = []
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            for scale in cfg['scales']:
                size = int(round(cfg['size'] * scale / 32) * 32)
                # Always resize the original batch: upstream mutates it across scales.
                x = F.interpolate(images, (size, size), mode='bilinear', align_corners=True) if size != cfg['size'] else images
                y = F.interpolate(masks, (size, size), mode='nearest') if size != cfg['size'] else masks
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = structure_loss(output(model, x), y)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg['clip'])
                scaler.step(opt); scaler.update()
                losses.append(loss.detach())
        val = evaluate(model, valset, device, cfg['batch_size'])
        score = val['normalized']['dice']
        if score > best:
            best, best_epoch = score, epoch
            torch.save(dict(model=model.state_dict(), config=cfg, epoch=epoch, manifest_sha256=provenance['manifest_sha256']), run_dir/'best.pt')
        if sched: sched.step()
        row = dict(epoch=epoch, loss=torch.stack(losses).mean().item(), val_dice=score,
                   val_iou=val['normalized']['iou'], seconds=time.perf_counter()-epoch_start, lr=opt.param_groups[0]['lr'])
        with (run_dir/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        ck = dict(model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                  scheduler=sched.state_dict() if sched else None, config=cfg, epoch=epoch,
                  best=best, best_epoch=best_epoch, elapsed=elapsed+time.perf_counter()-start,
                  manifest_sha256=provenance['manifest_sha256'], torch_rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                  python_rng=random.getstate(), numpy_rng=np.random.get_state())
        torch.save(ck, run_dir/'last.tmp'); (run_dir/'last.tmp').replace(run_dir/'last.pt')
        print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(run_dir/'best.pt', map_location=device, weights_only=False)['model'])
    testset = SegmentationDataset(manifest, 'test', cfg['size'])
    metrics = evaluate(model, testset, device, cfg['batch_size'], run_dir/'test')
    result = dict(status='completed', best_epoch=best_epoch, best_val_dice=best, epochs=cfg['epochs'],
                  elapsed_seconds=elapsed+time.perf_counter()-start, test=metrics,
                  checkpoint_sha256=sha256(run_dir/'best.pt'), manifest_sha256=provenance['manifest_sha256'])
    (run_dir/'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('--config', required=True); t.add_argument('--resume', action='store_true')
    e = sub.add_parser('evaluate'); e.add_argument('--checkpoint', required=True); e.add_argument('--manifest', required=True)
    e.add_argument('--output', required=True); e.add_argument('--split', default='test', choices=['val', 'test'])
    args = p.parse_args()
    if args.command == 'train': return train(args.config, args.resume)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model().to(device); model.load_state_dict(ck['model'])
    ds = SegmentationDataset(args.manifest, args.split, ck['config']['size'])
    dest = Path(args.output); dest.mkdir(parents=True, exist_ok=True)
    result = evaluate(model, ds, device, ck['config']['batch_size'], dest)
    (dest/'metrics.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result))

if __name__ == '__main__': main()
