"""Download the authors' split, retaining Drive IDs and resumable image files."""
import contextlib
import concurrent.futures
import json
import time
from pathlib import Path
import gdown
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/raw/clinicdb'
INDEX = ROOT / 'sources/clinicdb_drive_index.json'
URL = 'https://drive.google.com/drive/folders/1FPJr5f91uUCikxMvkwtZSEnYHemTZq1P'

def download(item):
    p = ROOT / item['path']
    if p.exists():
        try:
            if p.suffix == '.png':
                with Image.open(p) as im:
                    im.verify()
            return
        except Exception:
            pass
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            gdown.download(id=item['id'], output=str(p), quiet=True, use_cookies=False)
            if p.suffix == '.png':
                with Image.open(p) as im:
                    im.verify()
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

if __name__ == '__main__':
    if INDEX.exists():
        items = json.loads(INDEX.read_text())
    else:
        DEST.mkdir(parents=True, exist_ok=True)
        with open(ROOT / 'sources/drive_listing.log', 'w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                files = gdown.download_folder(URL, output=str(DEST), quiet=True, skip_download=True)
        items = [{'id': f.id, 'path': Path(f.local_path).relative_to(ROOT).as_posix()} for f in files]
        INDEX.write_text(json.dumps(items, indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for n, _ in enumerate(pool.map(download, items), 1):
            if n % 50 == 0 or n == len(items):
                print(f'ClinicDB: {n}/{len(items)} files verified', flush=True)
