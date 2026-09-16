import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screening.data import ROOT, validate_rows

def prepare(name):
    raw = ROOT / 'data/raw' / name
    rows, notes = [], []
    if name == 'clinicdb':
        for split in ('train', 'val', 'test'):
            folders = list(raw.rglob(f'{split}/images'))
            if len(folders) != 1:
                raise ValueError(f'Expected one {split}/images folder, got {folders}')
            folder = folders[0]
            for p in sorted(folder.glob('*.png')):
                rows.append(dict(id=p.stem, split=split, image=p.relative_to(ROOT).as_posix(),
                                 mask=(folder.parent / 'masks' / p.name).relative_to(ROOT).as_posix()))
        if len(rows) != 612:
            raise ValueError(f'Expected complete ClinicDB (612), got {len(rows)}')
        notes.append('Official split indexed from the MK-UNet README link; files may be restored from verified mirror. Frame-level split, patient/sequence independence unverified.')
    else:
        splits = yaml.safe_load((raw / 'train_test_split.yaml').read_text())
        overlap = set(splits['train']) & set(splits['test'])
        train = sorted(set(splits['train']) - overlap)
        random.Random(42).shuffle(train)
        val, train = set(train[:8]), set(train[8:])
        splits = dict(train=sorted(train), val=sorted(val), test=sorted(set(splits['test'])))
        notes += [f'Official split has overlapping IDs {sorted(overlap)}; removed from training, retained in test.',
                  'CWFID masks encode vegetation as black (0), soil as white (255); explicitly inverted for training.',
                  'Validation: eight images from remaining official training set, shuffle seed 42.',
                  'Binary vegetation masks merge crop and weed; no crop/weed discrimination claim.',
                  'Single-field benchmark: test images do not establish unseen-farm generalization.']
        for split, ids in splits.items():
            for i in ids:
                rows.append(dict(id=f'{i:03d}', split=split,
                                 foreground='black',
                                 image=f'data/raw/cwfid/images/{i:03d}_image.png',
                                 mask=f'data/raw/cwfid/masks/{i:03d}_mask.png'))
    validate_rows(rows)
    result = dict(dataset=name, counts=dict(Counter(r['split'] for r in rows)), notes=notes, samples=rows)
    dest = ROOT / 'data/manifests' / f'{name}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(name, result['counts'], notes)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=['clinicdb', 'cwfid', 'all'])
    a = p.parse_args()
    for name in (['clinicdb', 'cwfid'] if a.dataset == 'all' else [a.dataset]):
        prepare(name)
