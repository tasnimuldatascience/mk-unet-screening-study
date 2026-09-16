"""Restore unavailable Drive files from a mirror, verifying all overlapping pixels."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://www.kaggle.com/api/v1/datasets/download/balraj98/cvcclinicdb'

if __name__ == '__main__':
    archive = ROOT/'data/downloads/cvcclinicdb.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        with requests.get(URL,stream=True,timeout=60) as r:
            r.raise_for_status()
            with archive.open('wb') as f:
                for chunk in r.iter_content(1048576): f.write(chunk)
    items = json.loads((ROOT/'sources/clinicdb_drive_index.json').read_text())
    checked, restored = 0, 0
    with zipfile.ZipFile(archive) as z:
        for item in items:
            path = ROOT/item['path']
            if path.suffix != '.png': continue
            category = 'Original' if path.parent.name=='images' else 'Ground Truth'
            data = z.read(f'PNG/{category}/{path.name}')
            if path.exists():
                with Image.open(path) as a, Image.open(io.BytesIO(data)) as b:
                    if not np.array_equal(np.asarray(a.convert('RGB')), np.asarray(b.convert('RGB'))):
                        raise ValueError(f'Mirror pixels differ from author download: {path}')
                checked += 1
        # Do not write anything until the entire available overlap passes.
        for item in items:
            path = ROOT/item['path']
            if path.suffix != '.png' or path.exists(): continue
            category = 'Original' if path.parent.name=='images' else 'Ground Truth'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(z.read(f'PNG/{category}/{path.name}')); restored += 1
    audit=dict(url=URL,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               overlap_pixel_identical=checked,restored_files=restored,
               split_source='Official MK-UNet Drive index; filenames and split membership preserved')
    (ROOT/'sources/clinicdb_mirror_audit.json').write_text(json.dumps(audit,indent=2))
    print(audit)
