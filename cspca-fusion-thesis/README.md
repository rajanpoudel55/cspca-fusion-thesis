# Machine learning prediction of clinically significant prostate cancer from fused clinical and MRI features

Code that accompanies the thesis *Predictive Analysis for Early Detection of Prostate Cancer Using Clinical and Imaging Data* (Rajan Poudel, 2026).
The thesis fuses clinical variables with automatically extracted biparametric-MRI features, compares thirteen machine learning algorithms in three configurations
(fully automated, PI-RADS-informed, biopsy-pathway), and validates a locked model on an independent cohort.

## Contents
| Folder | What it holds | Thesis reference |
|---|---|---|
| `analysis/` | Fifteen numbered scripts (`00` to `14`) that build the derived tables, extract ResNet50 features, train and evaluate the models, and run the leakage audit, leave-one-centre-out validation, baselines, statistical tests, external validation and verification checks; a synthetic-data generator for smoke tests | Chapters 3 to 5; Appendices A to I; Table D.2 |
| `figures/` | Scripts that draw the data figures and diagrams of Chapters 2 to 6; `diagram_sources/` holds the draw.io and PlantUML source of Figure 2.3 | Figures in Chapters 2 to 6 |
| `notebooks/` | The Jupyter notebooks in which the experiments were run, with outputs; a cell map to the thesis sections (`notebooks/README.md`) | Chapters 3 and 4; Appendices A to I |
| `data/` | Place the file manifest here (see `data/README.md`) | Appendix I |
| `run_demo.py`, `RUN_GUIDE.md` | One-command synthetic-data demo; step-by-step guide to the real-data run | |

## Quick start
See [`RUN_GUIDE.md`](RUN_GUIDE.md) for the full guide: setup, the demo, the real-data run order, expected values and known caveats.
```bash
pip install -r requirements.txt            # Python 3.11
python analysis/00_environment_report.py   # compare versions with thesis Table 3.10
python run_demo.py                         # whole pipeline on SYNTHETIC data (about 7 minutes); log in demo_output/
```
The demo's numbers are meaningless clinically and will not match the thesis. `python run_demo.py --quick` skips two of the three model configurations.

To reproduce the reported results, download the public datasets and follow the run order in [`analysis/README.md`](analysis/README.md). That file lists the expected shapes and values for every step and the known deviations (thesis Appendix I).

## What the figure scripts do and do not do
`figures/make_ch4_figures.py` and `figures/make_ch56_figures.py` draw from the result values stored in `figures/ch4_data.py`, which are the values reported in the thesis. They do not recompute the results from raw data. Regenerate the figures with:
```bash
cd figures
python make_ch2_figures.py && python make_ch3_figures.py && python make_ch4_figures.py && python make_ch56_figures.py && python make_forest.py
```
Output goes to `figures/output/`.

## Data
- PI-CAI public training and development dataset: Zenodo, doi 10.5281/zenodo.6624726 (annotations and clinical information: github.com/DIAGNijmegen/picai_labels).
- Prostate-MRI-US-Biopsy (external cohort): The Cancer Imaging Archive, doi 10.7937/TCIA.2020.A61IOC1A.

## Licence and citation
MIT licence (see `LICENSE`).
Poudel, R. (2026). *Predictive Analysis for Early Detection of Prostate Cancer Using Clinical and Imaging Data* [Thesis]. Supervisor: Dr Nasim Ahmed.
