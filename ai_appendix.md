# AI Methodology Appendix
## ECON 3916 Final Project
**Cassandra Cinzori | Northeastern University | Spring 2026**

---

## Overview

This appendix documents my use of AI tools (Claude, Anthropic) throughout the ECON 3916 final project, following the P.R.I.M.E. framework: Prep → Request → Iterate → Mechanism Check → Evaluate. All AI-generated code was tested and run successfully. All AI-generated factual claims were verified against course materials or primary sources. Three detailed prompt-response pairs are included below.

---

## P.R.I.M.E. Pair 1: Dataset Exploration and Missing Data Strategy

**Prep**
Before prompting, I identified the dataset (Obesity_DataSet_2.csv, NHANES) and noted that it had 16 columns with varying rates of missingness. I reviewed Chapter 1 of the course materials on MCAR/MAR/MNAR and wanted to apply that framework systematically. My goal was to produce a defensible strategy for each column before writing any cleaning code.

**Request**
> "I have a dataset with the following missing value rates: Depressed (81%), Alcohol12PlusYr (12%), HHIncome (8.6%), TotChol (5.4%), BPSysAve (3.7%), Education (3.5%), MaritalStatus (3.3%), Smoke100 (3.3%), Height (0.8%), BMI_WHO (1.3% - this is my target), Diabetes (0.03%). Using the MCAR/MAR/MNAR framework from Chapter 1, classify each and recommend a handling strategy. The prediction target is BMI_WHO."

**Response (summarized)**
Claude classified each column and recommended: drop `Depressed` entirely (MNAR, 81% - imputing is not feasible); drop rows with missing `BMI_WHO` since it is the target; treat clinical measurements (BPSysAve, TotChol, Height) as MAR and apply median imputation; treat socioeconomic variables (HHIncome, Education) as MAR with a socioeconomic sensitivity caveat and apply mode imputation; treat Diabetes as MCAR (0.03%) and drop those 2 rows.

**Iterate**
I asked a follow-up: "Should I be worried about the socioeconomic features being MNAR instead of MAR - meaning people with lower income are more likely to not report income?" Claude confirmed this is a real concern in survey data, noted the distinction is empirically difficult to test without auxiliary data, and suggested acknowledging it as a limitation rather than trying to model it. I incorporated this nuance into the report's limitations section.

**Mechanism Check**
I verified the missing counts myself using `df.isnull().sum()` in Python. All percentages matched. I also confirmed the MCAR/MAR/MNAR definitions against Chapter 1 lecture notes. The `Depressed` column was indeed 81% missing - I confirmed this was not a data loading error by checking the raw CSV.

**Evaluate**
The MNAR concern about socioeconomic variables was a valuable nuance I would not have articulated as precisely on my own. However, I made the final judgment call on which columns to retain as features versus drop - Claude's suggestion to keep `Alcohol12PlusYr` despite 12% missingness was correct, given the mode imputation strategy. I also independently decided to drop `Depressed` rather than impute it, which Claude recommended and I agreed with.

---

## P.R.I.M.E. Pair 2: Random Forest Pipeline Construction

**Prep**
After fitting the Logistic Regression baseline, I needed to add a Random Forest model. I had used Random Forests in the class labs (Chapter 17–18) but had not built one inside a `ColumnTransformer + Pipeline` structure before. I wanted to avoid data leakage and ensure the preprocessing was consistent between the two models.

**Request**
> "I have a sklearn Pipeline with a ColumnTransformer that handles median imputation + scaling for numeric features and mode imputation + one-hot encoding for categorical features. I want to add a Random Forest classifier as a second model using the same preprocessor. What hyperparameters should I start with for a 4-class classification problem with ~4,700 observations and moderate class imbalance? Write the Pipeline code."

**Response (summarized)**
Claude recommended starting with `n_estimators=200`, `max_depth=15`, `min_samples_leaf=2`, and `n_jobs=-1`. It explained that `max_depth=15` balances complexity and overfitting risk, and `min_samples_leaf=2` prevents overfitting on the rare UnderWeight class. It provided the full Pipeline code including `random_state=42`.

**Iterate**
I asked: "Why `min_samples_leaf=2` rather than just leaving it at the default of 1?" Claude explained that with a rare class (UnderWeight, ~90 obs), `min_samples_leaf=1` can produce leaves that overfit to individual rare observations, while `min_samples_leaf=2` requires at least 2 samples per leaf and generalizes better. This connected directly to the bias-variance tradeoff from Chapter 6.

**Mechanism Check**
I ran the code, confirmed it executed without errors, and checked the output with `classification_report`. I also verified that both pipelines used the same `preprocessor` object - meaning the same imputation strategy - so the comparison was fair. I confirmed `n_jobs=-1` used all available CPU cores as expected.

**Evaluate**
The hyperparameter rationale was consistent with what I learned in class. However, I noted the Random Forest did not outperform Logistic Regression meaningfully (RF: 46.0% CV vs. LR: 45.1%), which Claude explained is not unusual when the dataset lacks strongly non-linear interactions. I made the independent decision to keep both models in the report and explain the similarity honestly, rather than artificially tuning to manufacture a larger gap - which would be misleading.

---

## P.R.I.M.E. Pair 3: SCR Report Structure

**Prep**
Before drafting the report, I reviewed Chapter 26 on the SCR (Situation-Complication-Resolution) framework and Spiegelhalter's guidance on communicating uncertainty. I wanted to make sure my recommendation section was honest about the model's limitations and did not overstate predictive performance.

**Request**
> "I'm writing the Results and Recommendation sections of a 5-page SCR report for a machine learning project. My Random Forest achieved 46% test accuracy on a 4-class classification problem (random baseline 25%, majority class baseline 35.4%). How should I frame these results honestly without underselling the model? And how do I write an actionable recommendation given moderate accuracy?"

**Response (summarized)**
Claude suggested framing the results relative to both baselines first - 46% is 21 percentage points above random and 11 points above the majority-class heuristic, which is meaningful. It recommended reporting the 95% CI from cross-validation alongside point estimates. For the recommendation, it suggested a risk-stratified framing: use the model as a screening tool that flags high-probability cases for follow-up, not as a definitive classifier.

**Iterate**
I asked: "The professor mentioned Spiegelhalter - should I reference uncertainty differently?" Claude explained that Spiegelhalter emphasizes communicating uncertainty at the individual prediction level, not just aggregate accuracy, which is why reporting confidence on individual predictions (via predicted probabilities) matters. I incorporated this into the Streamlit dashboard by showing predicted class probabilities and adding a bootstrap stability check.

**Mechanism Check**
I cross-checked the baselines against my own calculation: `y_test.value_counts().max() / len(y_test)` confirmed the majority class baseline as 35.4%. The 95% CI formula (mean ± 1.96 × std across CV folds) was confirmed against Chapter 15 notes on cross-validation confidence intervals.

**Evaluate**
The framing advice was consistent with Chapter 26 guidelines. I made the independent editorial decision to include a specific "Limitations" subsection in the recommendation rather than burying caveats in the methodology - this felt more honest and more useful to the stakeholder. I also independently decided to flag the missing body weight as the primary constraint on performance, which was a substantive judgment call Claude did not suggest unprompted.

---

## Summary

| Pair | AI Tool | Usage | Verified? | Human Judgment Applied |
|------|---------|-------|-----------|------------------------|
| 1 | Claude | Missing data MCAR/MAR/MNAR classification | Yes - checked against raw data and Ch. 1 notes | Final strategy decisions, MNAR caveat framing |
| 2 | Claude | Random Forest hyperparameter selection + Pipeline code | Yes - code run and output confirmed | Decision to retain both models and report similarity honestly |
| 3 | Claude | SCR report framing and uncertainty communication | Yes - baselines confirmed computationally, CI formula verified against Ch. 15 | Limitations section structure, weight-absence framing |

All AI-generated code was tested end-to-end before submission. All factual claims were source-checked against course materials, primary data, or independent calculation. Human judgment was applied at every decision point.
