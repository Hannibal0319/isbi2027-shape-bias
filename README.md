# ISBI 2027 submission: "Smaller looks rounder: correcting resolution-dependent bias of shape descriptors in bioimage and radiomics software"

Deadline: **Mon Oct 26, 2026, 23:59 New York time** (EDAS). 4 pages including references (a 5th page is allowed only for references/acknowledgements, at $200). Review is single-blind, so author names go on the paper.

- `paper/`: **the submission** (`main.tex`, `refs.bib`, `figures/fig_main.pdf`; build with `latexmk -pdf main.tex`).
- `shape/`: all code and results. `shape/pilots/` holds the feasibility scripts that are not used in the paper.
- `data/` (not in git, about 22 GB):
  - BBBC038 `stage1_train.zip` from https://data.broadinstitute.org/bbbc/BBBC038/, unzipped to `data/bbbc038/train`
  - PanNuke fold 1 from https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke, unzipped to `data/pannuke/fold1`
  - MSD Task06 lung from https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar, extracted to `data/msd/Task06_Lung`

  The large per-nucleus files (`shape/real_*`) are not in git; `extract_real.py` regenerates them.

## Main results
0. **Method (new): run-corrected Crofton estimator (RCC)**, `shape/runcorr.py`. Crofton estimators miss chords shorter than the sampling step h. For smooth boundaries the chord-length density is linear near 0, so the expected missed chords = n1/6, where n1 is the number of single-sample runs along that direction (second order: (19 n1 − 4 n2)/66). It is parameter-free and works in 2D and on anisotropic 3D grids. 2D circularity bias ≤0.005 at all radii ≥2 px (Crofton: 0.05 at r=2); 3D sphericity bias within ±0.005 for R≥5 mm up to 5 mm slices (Crofton: +0.046); lung phantoms at 5 mm: +0.026 (Crofton +0.042, PyRadiomics −0.136).
1. The default perimeter in scikit-image `regionprops` (and therefore CellProfiler FormFactor), OpenCV `arcLength` and ImageJ makes circularity depend on object size: +0.55 at r=2 px, 0 at r≈8, −0.08 at r=48.
2. Model: chain code gives P ≈ κ(L − 2√2), with κ = 8(√2−1)/π and δ = √2/π. Marching squares (PyRadiomics) gives κL; polygonised pixel edges give (4/π)L.
3. F1 is a post-hoc correction, P/κ + π (δ′ calibrated on synthetic shapes only: minimax 0.52, rounded to ½). With Cellpose re-segmentation it brings the resolution shift down from 0.7–3.3 SD to ≤0.05 SD.
4. PyRadiomics/IBSI sphericity of a sphere is 0.92 at 1 mm slices and 0.82 at 5 mm; resampling to isotropic makes it worse (0.80). F3, an anisotropic 13-direction Crofton estimator, stays at about 1.00.
5. PanNuke contrasts: the default inflates inflammatory-vs-neoplastic roundness from d≈0.35–0.40 to 0.67, flips connective-vs-neoplastic at 10×, and gives d=1.62 vs 0.35 (F1) with mixed scanners.
6. On lung-tumour phantoms, the PyRadiomics bias is −0.065 to −0.136 and F3's is +0.005 to +0.042.

## Environments
- System Python 3.14: numpy, scikit-image 0.26, OpenCV 4.13, scikit-learn, matplotlib.
- `.venv39` (Python 3.9): pyradiomics 3.0.1, installed with `uv pip install --no-build-isolation pyradiomics==3.0.1` after numpy<2, versioneer and setuptools.
- `.venvcp` (Python 3.11): cellpose<4 with CUDA torch (cu126 index).

## Reproduce (in `shape/`)
| step | command | output |
|---|---|---|
| synthetic 2D audit | `python exp1_synth.py`, then `../.venv39/Scripts/python prad_worker.py exp1_masks.npz exp1_prad.npy` | `exp1_synth.csv` (rerun `exp1_synth.py` once `exp1_prad.npy` exists to merge the PyRadiomics column) |
| synthetic 3D | `../.venv39/Scripts/python exp3d_synth.py` | `exp3d_synth.csv` |
| real nuclei, re-digitization | `python extract_real.py pannuke 2656`, `python extract_real.py bbbc 670`, then `prad_worker.py` on each `*_crops.npz` | `real_*.csv`, `real_*_prad.npy` |
| real nuclei, Cellpose re-segmentation | `../.venvcp/Scripts/python exp_reseg.py 10000 reseg_bbbc_all.csv`, `../.venvcp/Scripts/python exp_reseg_pn.py 1500 reseg_pn_1500.csv` | `reseg_*.csv` |
| lung phantoms | `../.venv39/Scripts/python exp3d_lung_phantom.py ../data/msd/Task06_Lung` | `exp3d_lung_phantom.csv` |
| RCC on real crops | `python add_rc_real.py` (adds `P_rc` to `real_*.csv`) | `real_*.csv` |
| Table 1 numbers | `python make_tables.py` | stdout |
| Table 2 (biological contrasts, classifier drop) | `python exp_contrast.py`; `python analyze_real2.py real_pannuke_2656.csv real_pannuke_2656_prad.npy` | stdout |
| Figure 1 (incl. schematic) | `python fig_main.py` | `../paper/figures/fig_main.pdf` |

Library code: `runcorr.py` (RCC, 2D/3D, any spacing), `estimators.py` (all 2D perimeter estimators, F1), `crofton3d.py` (plain spacing-aware 3D Crofton), `shapes.py` (synthetic shapes).

## Before submitting (TODO for the authors)
- Fill in the author names and affiliations in `paper/main.tex` (currently "Anonymous Author(s)"; ISBI review is single-blind).
- Make https://github.com/Hannibal0319/isbi2027-shape-bias public before submission (the paper links to it; it was created private).
- Check the AI-use disclosure in the Acknowledgments (ISBI requires one) and the conflict-of-interest/funding statement.
- Optional: open a scikit-image / CellProfiler issue proposing F1/Crofton as default (adds impact).
