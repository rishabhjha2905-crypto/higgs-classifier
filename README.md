# Higgs Boson Event Classifier

A machine learning project that classifies particle collision events from the CERN Large Hadron Collider as either Higgs boson signal or background noise, using data from the [2014 Kaggle Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson).

## Overview

The Higgs boson was confirmed experimentally in 2012 by the ATLAS and CMS detectors at CERN. A core challenge in that discovery was separating rare Higgs signal events from an overwhelming background of ordinary particle collisions — a classic signal-vs-noise classification problem.

This project replicates that classification task using gradient boosted trees on 30 physics-derived features from 250,000 simulated collision events, achieving 83.6% accuracy. The feature importance results closely mirror the real physics: the model learns to identify the Higgs primarily through its reconstructed invariant mass, exactly as the real detectors did.

## Dataset

- **Source:** [Kaggle Higgs Boson Challenge](https://www.kaggle.com/c/higgs-boson/data) (download manually)
- **Size:** 250,000 labeled events
- **Classes:** Signal/s (85,667) vs Background/b (164,333)
- **Features:** 30 measurements including transverse momentum, pseudorapidity, invariant mass, and missing energy

> **Note:** The dataset is not included in this repo due to file size. Download `training.csv` from the Kaggle link above and place it in the project folder before running.

## Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Background | 0.86 | 0.90 | 0.88 |
| Signal | 0.78 | 0.72 | 0.75 |
| **Overall** | | | **0.836** |

## Key Finding: Feature Importance

The two most important features were:

- **DER_mass_MMC (~0.34)** — the reconstructed Higgs candidate mass. The Higgs boson has a specific mass of ~125 GeV, so this feature essentially asks "does this event have the right mass to be a Higgs?" — exactly the discriminator real physicists use.
- **DER_mass_transverse_met_lep (~0.30)** — transverse mass involving missing energy, a key signature of the H→ττ decay channel where neutrinos escape the detector undetected.

Azimuthal angle features contributed almost nothing, consistent with the Higgs being produced isotropically.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (GradientBoostingClassifier)
- matplotlib, seaborn

## How to Run

1. Clone the repo
2. Download `training.csv` from [Kaggle](https://www.kaggle.com/c/higgs-boson/data) and place it in the project folder
3. Install dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
4. Open `higgs.ipynb` in VS Code or Jupyter and run all cells

## Project Structure

```
higgs-classifier/
├── higgs.ipynb       # Main notebook
├── .gitignore        # Excludes large data files
└── README.md
```
