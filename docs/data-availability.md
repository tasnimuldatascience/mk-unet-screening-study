# Data availability and fixed roles

All three datasets required by the proposed study have been acquired and audited locally. Raw images are excluded from Git because their original terms govern redistribution. The repository contains acquisition code, immutable split membership, hashes, and validation rules.

| Dataset | Available samples | Fixed role | Reproduction |
|---|---:|---|---|
| ClinicDB | 612 (489/61/62) | Medical source train/validation/test | `scripts/download_clinicdb.py`, `scripts/restore_clinicdb_mirror.py` |
| CVC-ColonDB | 380 | External medical test only | `scripts/prepare_colondb.py` |
| CWFID | 60 (31/8/21) | Independent agricultural feasibility branch | Upstream Git repository plus `scripts/prepare_data.py` |

The adaptation controller is calibrated only with synthetic shifts derived from source validation data. ColonDB labels are sealed from model selection, calibration, threshold selection, and adaptation; they are opened only for final offline scoring. The generated ColonDB manifest verifies pairing, dimensions, nonempty masks, archive and file SHA-256 hashes, and zero exact-image overlap with ClinicDB.

Sources: [ClinicDB paper](https://doi.org/10.1016/j.compmedimag.2015.02.007), [CVC-ColonDB archive](https://www.kaggle.com/datasets/longvil/cvc-colondb), and [CWFID repository](https://github.com/cwfid/dataset).
