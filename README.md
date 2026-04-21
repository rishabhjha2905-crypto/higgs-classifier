# Higgs Boson Event Classifier
 
A machine learning project that classifies particle collision events from the CERN Large Hadron Collider as either Higgs boson signal or background noise, using data from the [2014 Kaggle Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson).
 
## Overview
 
The Higgs boson was confirmed experimentally in 2012 by the ATLAS and CMS detectors at CERN. A core challenge in that discovery was separating rare Higgs signal events from an overwhelming background of ordinary particle collisions — a classic signal-vs-noise classification problem.
 
This project replicates that classification task using multiple models on 30 physics-derived features from 250,000 simulated collision events. Three approaches were tested and compared, with the best reaching 84.07% accuracy. The feature importance results closely mirror the real physics: the model learns to identify the Higgs primarily through its reconstructed invariant mass, exactly as the real detectors did.
 
## Dataset
 
- **Source:** [Kaggle Higgs Boson Challenge](https://www.kaggle.com/c/higgs-boson/data) (download manually)
- **Size:** 250,000 labeled events
- **Classes:** Signal/s (85,667) vs Background/b (164,333)
- **Features:** 30 measurements including transverse momentum, pseudorapidity, invariant mass, and missing energy
> **Note:** The dataset is not included in this repo due to file size. Download `training.csv` from the Kaggle link above and place it in the project folder before running.
 
## Model Comparison
 
Three approaches were tested and compared side by side:
 
| Model | Accuracy |
|-------|----------|
| Gradient Boosting (sklearn) | 83.62% |
| XGBoost | 84.06% |
| XGBoost + Missing Value Indicators | 84.07% |
 
The marginal gains between models reflect the inherent difficulty of the problem — the features are already heavily engineered by CERN physicists, leaving limited room for improvement with tree-based methods. Further gains would likely require a neural network approach.
 
## Key Finding: Feature Importance
 
The two most important features were:
 
- **DER_mass_MMC (~0.34)** — the reconstructed Higgs candidate mass. The Higgs boson has a specific mass of ~125 GeV, so this feature essentially asks "does this event have the right mass to be a Higgs?" — exactly the discriminator real physicists use.
- **DER_mass_transverse_met_lep (~0.30)** — transverse mass involving missing energy, a key signature of the H→ττ decay channel where neutrinos escape the detector undetected.
Azimuthal angle features contributed almost nothing, consistent with the Higgs being produced isotropically.
 
## Missing Value Handling
 
Several detector measurements are physically impossible for certain collision types and are recorded as -999. Two strategies were tested:
- **Median imputation** — replace -999 with the column median
- **Median + missing indicators** — additionally create binary columns flagging which measurements were absent, giving the model information about the missingness pattern itself
## Tech Stack
 
- Python
- pandas, NumPy
- scikit-learn (GradientBoostingClassifier)
- XGBoost
- matplotlib, seaborn
## How to Run
 
1. Clone the repo
2. Download `training.csv` from [Kaggle](https://www.kaggle.com/c/higgs-boson/data) and place it in the project folder
3. Install dependencies:
   ```
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```
4. Open `higgs.ipynb` in VS Code or Jupyter and run all cells
## Project Structure
 
```
higgs-classifier/
├── higgs.ipynb         # Main notebook
├── predictions.csv     # Model predictions on test data
├── .gitignore          # Excludes large data files
└── README.md
```
 