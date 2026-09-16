"""Acquire and audit CVC-ColonDB for external-test-only evaluation."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/longvil/cvc-colondb"
ARCHIVE = ROOT / "data/downloads/cvc-colondb.zip"
RAW = ROOT / "data/raw/colondb"
MANIFEST = ROOT / "data/manifests/colondb_external.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        urllib.request.urlretrieve(URL, ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as z:
        for kind in ("images", "masks"):
            out = RAW / kind
            out.mkdir(parents=True, exist_ok=True)
            for member in z.namelist():
                if member.startswith(f"CVC-ColonDB/{kind}/") and member.endswith(".png"):
                    (out / Path(member).name).write_bytes(z.read(member))

    images = {p.stem: p for p in (RAW / "images").glob("*.png")}
    masks = {p.stem: p for p in (RAW / "masks").glob("*.png")}
    if images.keys() != masks.keys() or len(images) != 380:
        raise RuntimeError(f"Expected 380 paired samples; got {len(images)} images/{len(masks)} masks")

    clinic = json.loads((ROOT / "data/manifests/clinicdb.json").read_text())
    clinic_hashes = {s["image_sha256"] for s in clinic["samples"]}
    samples, overlap = [], []
    for sid in sorted(images, key=int):
        image, mask = images[sid], masks[sid]
        with Image.open(image) as im, Image.open(mask) as ma:
            if im.size != ma.size:
                raise RuntimeError(f"Dimension mismatch for {sid}: {im.size} vs {ma.size}")
            if not ma.getbbox():
                raise RuntimeError(f"Empty mask: {sid}")
            size = list(im.size)
        ih = sha256(image)
        if ih in clinic_hashes:
            overlap.append(sid)
        samples.append({"id": sid, "split": "external_test", "image": image.relative_to(ROOT).as_posix(),
                        "mask": mask.relative_to(ROOT).as_posix(), "size": size,
                        "image_sha256": ih, "mask_sha256": sha256(mask)})
    if overlap:
        raise RuntimeError(f"ClinicDB/ColonDB image overlap detected: {overlap}")
    payload = {
        "dataset": "CVC-ColonDB", "role": "external_test_only", "count": len(samples),
        "source_url": URL, "archive_sha256": sha256(ARCHIVE),
        "protocol": "No model selection, calibration, or threshold tuning on these labels.",
        "cross_dataset_image_overlap_with_clinicdb": 0, "samples": samples,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Verified {len(samples)} external-test pairs; manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
