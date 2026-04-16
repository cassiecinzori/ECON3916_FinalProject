import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMI Weight Class Predictor",
    page_icon="🏥",
    layout="wide"
)

RANDOM_STATE = 42
NUMERIC_FEATURES = ['Age', 'BPSysAve', 'TotChol', 'Height']
CATEGORICAL_FEATURES = [
    'Gender', 'Race1', 'Education', 'HHIncome',
    'PhysActive', 'Smoke100', 'Diabetes',
    'Alcohol12PlusYr', 'MaritalStatus', 'Work'
]
CLASS_ORDER = ['UnderWeight', 'NormWeight', 'OverWeight', 'Obese']
CLASS_COLORS = {
    'UnderWeight': '#4C9BE8',
    'NormWeight':  '#5CB85C',
    'OverWeight':  '#F0AD4E',
    'Obese':       '#D9534F'
}

# ── Load or train model ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    df = pd.read_csv('Obesity_DataSet_2.csv')
    df = df.drop(columns=['Depressed']).dropna(subset=['BMI_WHO', 'Diabetes']).drop_duplicates()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['BMI_WHO']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), NUMERIC_FEATURES),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), CATEGORICAL_FEATURES)
    ])

    lr = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE))
    ])
    rf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200, max_depth=15,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # CV scores for display
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    lr_cv = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
    rf_cv = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

    return lr, rf, lr_cv, rf_cv, X_train.columns.tolist()


# ── UI ────────────────────────────────────────────────────────────────────────
st.title(" BMI Weight Class Predictor")
st.markdown(
    "Predict an individual's BMI weight class from behavioral, demographic, and "
    "clinical health indicators. Based on NHANES data (CDC)."
)
st.caption(
    "️**Predictive model, not diagnostic tool.** "
    "Feature importance reflects predictive associations, not causal effects. "
    "This tool is for research and educational purposes only."
)

st.divider()

with st.spinner("Loading models..."):
    lr_model, rf_model, lr_cv_scores, rf_cv_scores, _ = load_models()

# ── Sidebar: inputs ───────────────────────────────────────────────────────────
st.sidebar.header("Patient Inputs")
st.sidebar.markdown("Adjust the inputs below to generate a prediction.")

model_choice = st.sidebar.radio(
    "Select Model",
    ["Random Forest", "Logistic Regression"],
    help="Random Forest has lower variance across folds (tighter CI)."
)

st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age", min_value=18, max_value=80, value=40, step=1)
gender = st.sidebar.selectbox("Gender", ["female", "male"])
race = st.sidebar.selectbox("Race/Ethnicity", ["White", "Black", "Mexican", "Hispanic", "Other"])
education = st.sidebar.selectbox(
    "Education Level",
    ["8th Grade", "9 - 11th Grade", "High School", "Some College", "College Grad"]
)
hhincome = st.sidebar.selectbox(
    "Household Income",
    ["0-4999", "5000-9999", "10000-14999", "15000-19999",
     "20000-24999", "25000-34999", "35000-44999", "45000-54999",
     "55000-64999", "65000-74999", "75000-99999", "more 99999"],
    index=5
)
marital = st.sidebar.selectbox(
    "Marital Status",
    ["Married", "NeverMarried", "Divorced", "LivePartner", "Widowed", "Separated"]
)
work = st.sidebar.selectbox("Work Status", ["Working", "NotWorking", "Looking"])

st.sidebar.subheader("Lifestyle Behaviors")
phys_active = st.sidebar.radio("Physically Active?", ["Yes", "No"])
smoke100 = st.sidebar.radio("Smoked 100+ cigarettes lifetime?", ["No", "Yes"])
alcohol = st.sidebar.radio("Had 12+ drinks in a year?", ["Yes", "No"])
diabetes = st.sidebar.radio("Diabetes diagnosis?", ["No", "Yes"])

st.sidebar.subheader("Clinical Measurements")
height = st.sidebar.slider("Height (cm)", min_value=135.0, max_value=200.0, value=170.0, step=0.5)
bp = st.sidebar.slider("Systolic Blood Pressure (mmHg)", min_value=78, max_value=200, value=120, step=1)
chol = st.sidebar.slider("Total Cholesterol (mmol/L)", min_value=1.5, max_value=13.0, value=5.0, step=0.1)

# ── Build input row ───────────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    'Age': age,
    'BPSysAve': bp,
    'TotChol': chol,
    'Height': height,
    'Gender': gender,
    'Race1': race,
    'Education': education,
    'HHIncome': hhincome,
    'PhysActive': phys_active,
    'Smoke100': smoke100,
    'Diabetes': diabetes,
    'Alcohol12PlusYr': alcohol,
    'MaritalStatus': marital,
    'Work': work
}])

model = rf_model if model_choice == "Random Forest" else lr_model
prediction = model.predict(input_data)[0]
proba = model.predict_proba(input_data)[0]
proba_dict = dict(zip(model.classes_, proba))

# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Predicted BMI Class")
    color = CLASS_COLORS.get(prediction, '#888')
    confidence = proba_dict[prediction]

    st.markdown(
        f"<div style='background-color:{color}22; border-left: 5px solid {color}; "
        f"padding: 16px 20px; border-radius: 6px; margin-bottom: 10px;'>"
        f"<span style='font-size:2rem; font-weight:700; color:{color}'>{prediction}</span>"
        f"<br><span style='font-size:0.95rem; color:#555;'>Model confidence: "
        f"<b>{confidence*100:.1f}%</b></span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Confidence interval note
    n_boot = 200
    boot_preds = []
    np.random.seed(RANDOM_STATE)
    for _ in range(n_boot):
        noise = input_data.copy()
        noise['Age'] = age + np.random.randint(-1, 2)
        noise['BPSysAve'] = bp + np.random.randint(-3, 4)
        boot_preds.append(model.predict(noise)[0])

    boot_series = pd.Series(boot_preds)
    boot_counts = boot_series.value_counts(normalize=True)
    top_boot = boot_counts.index[0]
    top_pct = boot_counts.iloc[0]

    st.caption(
        f"Bootstrap stability check (n=200 perturbations): "
        f"**{top_boot}** predicted {top_pct*100:.0f}% of the time. "
        f"Uncertainty is higher near class boundaries."
    )

    st.subheader("Model Performance")
    cv_scores = rf_cv_scores if model_choice == "Random Forest" else lr_cv_scores
    cv_mean = cv_scores.mean()
    cv_ci = 1.96 * cv_scores.std()
    st.metric(
        label=f"{model_choice} - 5-Fold CV Accuracy",
        value=f"{cv_mean*100:.1f}%",
        delta=f"±{cv_ci*100:.1f}% (95% CI)"
    )
    st.caption("Random baseline: 25.0% | Majority class baseline: 35.4%")

with col2:
    st.subheader("Predicted Class Probabilities")

    # Build ordered proba for all 4 classes
    ordered_proba = [proba_dict.get(cls, 0) for cls in CLASS_ORDER]
    bar_colors = [CLASS_COLORS[cls] for cls in CLASS_ORDER]
    highlight = [1.0 if cls == prediction else 0.6 for cls in CLASS_ORDER]
    final_colors = [
        f"#{int(int(c[1:3],16)*h):02x}{int(int(c[3:5],16)*h):02x}{int(int(c[5:7],16)*h):02x}"
        for c, h in zip(bar_colors, highlight)
    ]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    bars = ax.barh(CLASS_ORDER, ordered_proba, color=bar_colors, edgecolor='white',
                   linewidth=1.0, alpha=0.85)
    for bar, val, cls in zip(bars, ordered_proba, CLASS_ORDER):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', fontsize=10,
                fontweight='bold' if cls == prediction else 'normal')
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Probability', fontsize=10)
    ax.set_title(f'Predicted probabilities - {model_choice}', fontsize=10, fontweight='bold')
    ax.axvline(x=0.25, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(0.25, -0.6, 'Random\nbaseline', ha='center', fontsize=7, color='gray')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── About section ─────────────────────────────────────────────────────────────
with st.expander("About this model"):
    st.markdown("""
**Dataset:** NHANES (National Health and Nutrition Examination Survey), CDC  
**N:** ~4,761 observations after deduplication  
**Target:** BMI weight class (UnderWeight / NormWeight / OverWeight / Obese)  
**Train/test split:** 80/20, stratified, random_state=42  

| Model | CV Accuracy | 95% CI |
|-------|------------|--------|
| Random Forest | ~46.0% | ±2.4% |
| Logistic Regression | ~45.1% | ±4.3% |
| Majority class baseline | 35.4% | - |
| Random baseline | 25.0% | - |

**Note on performance:** This dataset does not include body weight - BMI is derived from height and weight, so predicting BMI category without weight is inherently constrained. Both models substantially outperform the random and majority-class baselines.

 **All feature importance values reflect predictive associations, not causal effects.**  
This tool is for ECON 3916 educational purposes only. Not a clinical diagnostic tool.
    """)

st.caption(
    "ECON 3916 Final Project | Cassandra Cinzori | Northeastern University | Spring 2026"
)
