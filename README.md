# ECON 3916: ML Prediction Project
**Predicting BMI Weight Class from Behavioral and Clinical Health Indicators**  
Spring 2026 | Cassandra Cinzori | Northeastern University

---

## Project Overview

This project applies supervised machine learning to predict an individual's BMI weight class (Underweight, Normal Weight, Overweight, or Obese) using behavioral, demographic, and clinical features from the NHANES dataset.

**Prediction question:** Can we predict BMI weight class from routine health intake data?  
**Stakeholder:** Primary care providers flagging patients for weight-related interventions  
**Models:** Logistic Regression (baseline) vs. Random Forest  
**Target variable:** `BMI_WHO` - four-class classification

> **Note:** This is a prediction problem, not causal inference. Feature importance scores reflect predictive associations, not causal effects.

---

## Repository Structure

```
ECON3916_FinalProject/
├── ECON3916_FinalProject_Checkpoint.ipynb   # Checkpoint: proposal, EDA, baseline model
├── Obesity_DataSet_2.csv                    # NHANES dataset
├── app.py                                   # Streamlit dashboard (final submission)
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## Dataset

- **Source:** National Health and Nutrition Examination Survey (NHANES), CDC
- **URL:** https://www.cdc.gov/nchs/nhanes/index.htm
- **Accessed:** April 2026
- **Observations:** 7,481 raw (4,761 after deduplication and dropping missing targets)
- **Features:** 14 input features - demographics, socioeconomics, lifestyle behaviors, clinical indicators
- **Target:** `BMI_WHO` (UnderWeight / NormWeight / OverWeight / Obese)

---

## Reproducing the Analysis

### 1. Clone the repo
```bash
git clone https://github.com/cassiecinzori/ECON3916_FinalProject.git
cd ECON3916_FinalProject
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `ECON3916_FinalProject_Checkpoint.ipynb` in Jupyter or Google Colab and run all cells. The dataset (`Obesity_DataSet_2.csv`) must be in the same directory as the notebook.

To open in Colab directly, click the **Open in Colab** badge at the top of the notebook on GitHub.

### 4. Launch the Streamlit app (final submission)
```bash
streamlit run app.py
```

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
streamlit>=1.31.0
```

See `requirements.txt` for pinned versions.

---

## Results (Checkpoint)

| Model | Test Accuracy |
|-------|--------------|
| Logistic Regression (baseline) | ~46% |
| Random Forest (final) | TBD |

Accuracy is moderate given four classes and class imbalance (UnderWeight = ~2% of data). Per-class precision/recall reported in the notebook.

---

## AI Usage

AI tools (Claude) were used throughout this project and are documented in the AI Methodology Appendix (submitted separately as PDF) using the P.R.I.M.E. framework per course requirements.
