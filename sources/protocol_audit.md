# Protocol audit

The requested source is `Student_Screening_Guidelines.pdf` (2 pages). Read in full, including visual inspection.

## Selected work

MK-UNet, Rahman and Marculescu, ICCV Workshops 2025.
Paper: https://arxiv.org/html/2509.18493v1
Official implementation: https://github.com/SLDGroup/MK-UNet
Pinned revision: 2fa8b230a057602539af7655203ad434bf56a5b6.
The unmodified upstream model is imported directly; BSD-3-Clause license retained.

Paper Table 1: standard MK-UNet ClinicDB Dice 93.48%, averaged over five runs.
Paper section 4.2: 200 epochs, AdamW lr=weight_decay=0.0001, batch 16,
352 pixels for polyps, three scales (0.75,1,1.25), clip 0.5, no augmentation.
Current code instead defaults to lr 0.0005, batch 8, cosine schedule, augmentation.
We use paper settings for the reference configuration, one seed (42).
One seed is a reference replication attempt, not reproduction of the five-run mean.
The ClinicDB run uses CUDA float16 autocast with gradient scaling to fit batch 16
on the 8 GB laptop GPU. An FP32 feasibility check exceeded the practical device-memory
budget. CWFID uses FP32. Precision is recorded in
the configuration and is an additional deviation from the historical runtime.

## Corrections in the independent runner

The Drive transfer retrieved 454 image/mask files. A public archive from
https://www.kaggle.com/datasets/balraj98/cvcclinicdb supplied the remaining 770 files.
All 454 overlapping files were pixel-identical. Official Drive split membership
was retained; archive SHA-256 and counts are in `clinicdb_mirror_audit.json`.

- Upstream PIL `size` is width,height; test code interprets it as height,width.
  The new runner resizes predictions directly to original mask height,width.
- Upstream rescales `images` in place through its scale loop; the nominal 1.0 pass
  therefore receives the preceding 0.75 tensor. Each scale now starts from the original batch.
- Ground truth is evaluated at its native resolution, without down/up sampling it.
- Files are paired by basename rather than merely zipping sorted arrays.
- Test data are evaluated only after selecting the best validation checkpoint.
- Per-image normalized 0.5 predictions follow upstream postprocessing; fixed sigmoid
  0.5 metrics are additionally reported because min-max normalization destroys calibration.
- Current PyTorch/CUDA required for the laptop differs from the paper's old runtime.

## New domain

CWFID: https://github.com/cwfid/dataset (revision 36290d0912a032f0bd1a5678ed55457ddafde217).
Dataset paper: https://doi.org/10.1007/978-3-319-16220-1_8.
Non-commercial research use only; citation retained. Binary vegetation task.
Official train/test lists overlap at image 028. Removed from training; retained in test.
Validation drawn only from remaining training pool using seed 42. Counts: 31/8/21.
Masks use BLACK vegetation and WHITE background, confirmed visually against RGB and
red/green plant annotations. The loader explicitly inverts CWFID masks, and a
regression test checks the resulting foreground fraction.
Independent retraining on this new task meets the PDF's experiment-2 specification;
it does not establish zero-shot transfer or unseen-farm domain generalization.

## Interpretation constraints

ClinicDB split is frame-level; patient/sequence independence is unverified.
CWFID is small and from one field. Image bootstrap intervals ignore correlated acquisition.
No clinical suitability, field deployment readiness, or statistically established superiority is claimed.
