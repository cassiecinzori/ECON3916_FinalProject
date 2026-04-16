# Predicting BMI Weight Class from Behavioral and Clinical Health Indicators
## ECON 3916 Final Project Report
**Cassandra Cinzori | Northeastern University | Spring 2026**

---

## 1. Problem Statement

Obesity and weight-related illness represent a leading public health challenge in the United States. Early identification of individuals at elevated risk allows providers to intervene before weight-related complications develop — but formal BMI computation requires physical measurement and is often not the first step in a clinical encounter.

**Prediction question:** Can we predict an individual's BMI weight class (Underweight, Normal Weight, Overweight, or Obese) from behavioral, demographic, and clinical health indicators available at a routine medical intake?

This is a **prediction problem, not a causal inference problem.** The model identifies patterns in observable features that are associated with BMI category — it does not claim that any feature causally drives weight outcomes.

**Stakeholder:** Primary care providers. The model would help a clinician or intake coordinator flag patients for weight-related health discussions based on information gathered before a physical exam is complete, using features like age, physical activity level, blood pressure, and socioeconomic status. This enables earlier, more targeted outreach.

**Decision enabled:** Which patients should be prioritized for weight-related intervention at a given intake visit?

---

## 2. Data Description

**Source:** National Health and Nutrition Examination Survey (NHANES), conducted by the CDC. NHANES is a nationally representative survey combining interviews and physical examinations. URL: https://www.cdc.gov/nchs/nhanes/index.htm. Accessed April 2026.

**Size:** 7,481 raw observations. After removing 2,646 exact duplicate rows (a known artifact of NHANES's complex survey oversampling design), 97 rows with missing target values, and 2 rows with missing diabetes status, the working dataset contains 4,761 observations.

**Features:** 14 input variables across four categories:

- *Demographics:* Age (continuous, 18–80), Gender (binary), Race/Ethnicity (5 categories)
- *Socioeconomic:* Education (5 levels), Household Income (12 brackets), Marital Status, Work Status
- *Lifestyle behaviors:* Physical Activity (binary), Smoking history (binary), Alcohol use (binary), Diabetes diagnosis (binary)
- *Clinical indicators:* Systolic Blood Pressure (continuous), Total Cholesterol (continuous), Height (continuous)

**Target variable:** `BMI_WHO` — four-class ordinal label: UnderWeight, NormWeight, OverWeight, Obese. Class distribution is moderately imbalanced: Obese (35.3%), OverWeight (33.8%), NormWeight (28.9%), UnderWeight (1.9%).

**Quality assessment:** Missing data was assessed using the MCAR/MAR/MNAR framework. The `Depressed` column (81% missing) was dropped. Remaining missingness — ranging from 0.8% (Height) to 12.0% (Alcohol12PlusYr) — was assessed as likely MAR and handled via median imputation for continuous features and mode imputation for categorical features, fit only on training data to prevent leakage.

A key limitation of the dataset is the absence of body weight. BMI is mathematically computed from height and weight, so predicting BMI category without weight is inherently constrained. This represents the performance ceiling for any model trained on these features.

---

## 3. Methodology

**Train/test split:** 80/20 stratified split with `random_state=42`, preserving class distribution in both sets. Training set: 3,808 observations. Test set: 953 observations.

**Preprocessing pipeline:** All preprocessing was performed inside a `sklearn` Pipeline to prevent data leakage. Numeric features received median imputation followed by standard scaling. Categorical features received mode imputation followed by one-hot encoding with unknown-category handling. Imputers were fit exclusively on training data.

**Models compared:**

*Model 1 — Logistic Regression (baseline):* Multinomial logistic regression using the `lbfgs` solver with `max_iter=1000`. Selected as baseline because it is interpretable, fast to train, and provides a linear decision boundary against which to compare more complex models. No hyperparameter tuning beyond solver selection.

*Model 2 — Random Forest:* Ensemble of 200 decision trees with `max_depth=15` and `min_samples_leaf=2`. Selected because Random Forests capture non-linear feature interactions and tend to be robust to moderate class imbalance. Hyperparameters were chosen based on model complexity considerations: `max_depth=15` is deep enough to capture interactions without overfitting, and `min_samples_leaf=2` prevents overfitting on rare classes.

**Cross-validation:** 5-fold stratified cross-validation was used to evaluate generalization. Stratification ensured the UnderWeight minority class appeared in each fold. Cross-validation scores were used to compute 95% confidence intervals on accuracy (mean ± 1.96 × standard deviation across folds).

**Evaluation metrics:** Accuracy, balanced accuracy, and per-class precision, recall, and F1-score. Balanced accuracy is reported alongside overall accuracy to account for class imbalance. Precision and recall are reported per class because a misclassification of Obese as NormWeight is clinically more serious than misclassifying OverWeight as Obese.

---

## 4. Results

**Model performance:**

| Model | CV Accuracy | 95% CI | Test Accuracy | Balanced Accuracy |
|-------|------------|--------|--------------|-------------------|
| Random baseline | 25.0% | — | — | — |
| Majority class baseline | 35.4% | — | — | — |
| Logistic Regression | 45.1% | ±4.3% | 46.3% | 34.0% |
| Random Forest | 46.0% | ±2.4% | 46.2% | 33.9% |

Both models substantially outperform the random baseline (25%) and the majority-class baseline (35.4%). The Random Forest shows lower variance across folds (±2.4% vs ±4.3%), indicating more stable predictions. Overall accuracy is similar, but the Random Forest is preferred for deployment due to its consistency.

**Per-class performance (Random Forest):** NormWeight and Obese are predicted with reasonable precision (~51% and ~46% respectively). OverWeight is harder to distinguish — it sits in the middle of the ordinal scale and shares characteristics with both adjacent classes. UnderWeight recall is near zero due to severe class imbalance (only 90 observations out of 4,761).

**Feature importance:** ⚠️ *The following reflects predictive association, not causal effect.* The four strongest predictors are the continuous clinical and demographic features: Age, Systolic Blood Pressure, Total Cholesterol, and Height each contribute roughly 11–13% of total importance. Categorical behavioral features (Education, Income, Race, Smoking) each contribute 2–3%, suggesting that while they are informative in aggregate, no single behavioral feature dominates.

**Uncertainty:** Confidence intervals on CV accuracy are reported above. The model's confidence scores on individual predictions should be interpreted cautiously — particularly for inputs near class boundaries (e.g., borderline OverWeight vs. Obese inputs), where the model's predicted probabilities are more evenly distributed. The bootstrap stability check in the Streamlit dashboard provides a practical sense of prediction uncertainty at the individual level.

---

## 5. Recommendation

**Actionable recommendation:** Deploy the Random Forest model as a screening tool within primary care intake workflows. Patients predicted as Obese or OverWeight with model confidence above 50% should be flagged for a weight-related health conversation and a formal BMI measurement. Given the model's ~46% overall accuracy, the tool should be used to prioritize attention, not replace clinical judgment.

**Uncertainty bounds:** The model's 5-fold CV accuracy of 46.0% (±2.4%, 95% CI: 43.6%–48.4%) means it correctly classifies roughly 4–5 out of every 10 patients. This is meaningfully better than chance (25%) or defaulting to the majority class (35.4%), but it is not a high-precision instrument. Clinicians should treat predictions as probabilistic signals.

**Limitations:**
- The absence of body weight is the primary constraint on model performance. A version of this tool that includes self-reported or estimated weight would substantially improve accuracy.
- UnderWeight is predicted poorly due to its rarity in the dataset. The tool should not be used to screen for underweight risk.
- NHANES is a nationally representative U.S. sample. Performance may degrade when applied to populations with different demographic compositions.
- Feature importance reflects correlation in this dataset, not causal mechanisms. Clinical decisions should not be made based on importance rankings alone.

**Next steps:** (1) Collect or augment data with self-reported weight to improve predictive ceiling. (2) Apply SMOTE or class-weighting to improve UnderWeight recall. (3) Explore ordinal classification approaches that exploit the natural ordering of BMI categories (UnderWeight < NormWeight < OverWeight < Obese), which standard multiclass models ignore.

---

*Word count: ~900 words (within 5-page target). All feature importance findings are predictive associations only — not causal claims.*
