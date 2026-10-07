# RIDER Lung CT (coffee-break) pipeline — shared by both ISBI 2027 papers

Data: TCIA collection "RIDER Lung CT" (Zhao et al., Sci Data 2024), 32 patients, same-day test/retest, each scan reconstructed at 1.25/2.5/5 mm with LUNG (sharp) and STANDARD (smooth) kernels, radiologist DICOM SEG per reconstruction.
Derived per-tumour values are not redistributed (dataset licence); only the scripts and aggregate summaries are included.

1. Download into `data/rider/`: `series.json` from the NBIA API (`getSeries?Collection=RIDER Lung CT`), the "Tumor Segmentation" SEG series into `seg/`, and the CT series they reference into `ct/` (NBIA `getImage`; about 14 GB compressed).
2. Copy these scripts to `data/rider/` and run (Python 3.9 venv with pyradiomics, pydicom, pydicom-seg, SimpleITK):
   `python convert.py` (CT + SEG to NIfTI, index), `python qc.py` (slice-spacing check; non-uniform series excluded),
   then assign visits by acquisition-time clustering (see the session notes in the paper-1 repo README),
   `python rider_shape.py` (paper 1: sphericity per reconstruction), `python rider_texture.py` (paper 2: soft-tissue internal reference + tumour texture),
   `python rider_analyze.py` (all numbers used in both papers).
