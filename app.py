import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMI Weight Class Predictor",
    page_icon=":hospital:",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] {
    background-color: #f8f9fb;
    border-right: 1px solid #e8eaed;
}
.sidebar-section {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9ca3af;
    margin: 1.4rem 0 0.5rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #e5e7eb;
}
.metric-row { display: flex; gap: 0.8rem; margin: 1rem 0; }
.metric-card {
    flex: 1;
    background: #f8f9fb;
    border: 1px solid #e8eaed;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9ca3af;
    margin-bottom: 4px;
}
.metric-value { font-size: 1.4rem; font-weight: 700; color: #111827; }
.metric-sub { font-size: 0.75rem; color: #6b7280; margin-top: 2px; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
NUMERIC_FEATURES = ['Age', 'BPSysAve', 'TotChol', 'Height']
CATEGORICAL_FEATURES = [
    'Gender', 'Race1', 'Education', 'HHIncome',
    'PhysActive', 'Smoke100', 'Diabetes',
    'Alcohol12PlusYr', 'MaritalStatus', 'Work'
]
CLASS_ORDER  = ['UnderWeight', 'NormWeight', 'OverWeight', 'Obese']
CLASS_LABELS = {'UnderWeight': 'Under Weight', 'NormWeight': 'Normal Weight',
                'OverWeight': 'Over Weight', 'Obese': 'Obese'}
CLASS_COLORS = {'UnderWeight': '#3b82f6', 'NormWeight': '#22c55e',
                'OverWeight': '#f59e0b', 'Obese': '#ef4444'}

MARITAL_DISPLAY = {
    'Married': 'Married', 'Never Married': 'NeverMarried',
    'Divorced': 'Divorced', 'Living with Partner': 'LivePartner',
    'Widowed': 'Widowed', 'Separated': 'Separated'
}
WORK_DISPLAY = {'Working': 'Working', 'Not Working': 'NotWorking', 'Looking for Work': 'Looking'}
INCOME_DISPLAY = {
    '$0 - $4,999': '0-4999', '$5,000 - $9,999': '5000-9999',
    '$10,000 - $14,999': '10000-14999', '$15,000 - $19,999': '15000-19999',
    '$20,000 - $24,999': '20000-24999', '$25,000 - $34,999': '25000-34999',
    '$35,000 - $44,999': '35000-44999', '$45,000 - $54,999': '45000-54999',
    '$55,000 - $64,999': '55000-64999', '$65,000 - $74,999': '65000-74999',
    '$75,000 - $99,999': '75000-99999', '$100,000 or more': 'more 99999'
}
EDUCATION_DISPLAY = {
    '8th Grade or Less': '8th Grade', '9th - 11th Grade': '9 - 11th Grade',
    'High School / GED': 'High School', 'Some College': 'Some College',
    'College Graduate': 'College Grad'
}
RACE_DISPLAY = {
    'White': 'White', 'Black / African American': 'Black',
    'Mexican American': 'Mexican', 'Hispanic': 'Hispanic', 'Other': 'Other'
}

# ── Load / train models ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on NHANES data - about 30 seconds...")
def load_models():
    df = pd.read_csv('Obesity_DataSet_2.csv')
    df = df.drop(columns=['Depressed']).dropna(subset=['BMI_WHO', 'Diabetes']).drop_duplicates()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['BMI_WHO']
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), NUMERIC_FEATURES),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('encoder', OneHotEncoder(handle_unknown='ignore',
                                                    sparse_output=False))]), CATEGORICAL_FEATURES)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    rf = Pipeline([('preprocessor', preprocessor),
                   ('classifier', RandomForestClassifier(
                       n_estimators=100, max_depth=15,
                       min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1))])
    lr = Pipeline([('preprocessor', preprocessor),
                   ('classifier', LogisticRegression(
                       solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE))])
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_cv = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    lr_cv = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
    return rf, lr, rf_cv, lr_cv

rf_model, lr_model, rf_cv_scores, lr_cv_scores = load_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## BMI Weight Class Predictor")
st.markdown(
    "Predict an individual's BMI weight class from behavioral, demographic, and "
    "clinical health indicators using NHANES data (CDC)."
)
st.caption(
    "Predictive model only - not a diagnostic tool. "
    "Feature importance reflects predictive associations, not causal effects. "
    "For research and educational purposes only."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Inputs")
    model_choice = st.radio("Model", ["Random Forest", "Logistic Regression"],
                            help="Random Forest shows lower variance across CV folds.")

    st.markdown('<div class="sidebar-section">Demographics</div>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 80, 40)

    gender_display = st.selectbox("Gender", ["Female", "Male"])
    gender = gender_display.lower()

    race_display = st.selectbox("Race / Ethnicity", list(RACE_DISPLAY.keys()))
    race = RACE_DISPLAY[race_display]

    education_display = st.selectbox("Education Level", list(EDUCATION_DISPLAY.keys()))
    education = EDUCATION_DISPLAY[education_display]

    income_display = st.selectbox("Household Income", list(INCOME_DISPLAY.keys()), index=5)
    hhincome = INCOME_DISPLAY[income_display]

    marital_display = st.selectbox("Marital Status", list(MARITAL_DISPLAY.keys()))
    marital = MARITAL_DISPLAY[marital_display]

    work_display = st.selectbox("Work Status", list(WORK_DISPLAY.keys()))
    work = WORK_DISPLAY[work_display]

    st.markdown('<div class="sidebar-section">Lifestyle Behaviors</div>', unsafe_allow_html=True)
    phys_active = st.radio("Physically active?",                   ["Yes", "No"], horizontal=True)
    smoke100    = st.radio("Smoked 100+ cigarettes (lifetime)?",   ["Yes", "No"], horizontal=True)
    alcohol     = st.radio("12 or more drinks in a year?",         ["Yes", "No"], horizontal=True)
    diabetes    = st.radio("Diabetes diagnosis?",                  ["Yes", "No"], horizontal=True)

    st.markdown('<div class="sidebar-section">Clinical Measurements</div>', unsafe_allow_html=True)
    height = st.slider("Height (cm)",                     135.0, 200.0, 170.0, 0.5)
    bp     = st.slider("Systolic Blood Pressure (mmHg)",  78,    200,   120,   1)
    chol   = st.slider("Total Cholesterol (mmol/L)",      1.5,   13.0,  5.0,   0.1)

# ── Build input DataFrame ─────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    'Age': age, 'BPSysAve': bp, 'TotChol': chol, 'Height': height,
    'Gender': gender, 'Race1': race, 'Education': education,
    'HHIncome': hhincome, 'PhysActive': phys_active, 'Smoke100': smoke100,
    'Diabetes': diabetes, 'Alcohol12PlusYr': alcohol,
    'MaritalStatus': marital, 'Work': work
}])

model      = rf_model if model_choice == "Random Forest" else lr_model
cv_scores  = rf_cv_scores if model_choice == "Random Forest" else lr_cv_scores
prediction = model.predict(input_data)[0]
proba      = model.predict_proba(input_data)[0]
proba_dict = dict(zip(model.classes_, proba))
confidence = proba_dict[prediction]
cv_mean    = cv_scores.mean()
cv_ci      = 1.96 * cv_scores.std()
color      = CLASS_COLORS[prediction]
label      = CLASS_LABELS[prediction]

# Bootstrap stability
np.random.seed(RANDOM_STATE)
boot_preds = []
for _ in range(200):
    noise = input_data.copy()
    noise['Age']      = age + np.random.randint(-1, 2)
    noise['BPSysAve'] = bp  + np.random.randint(-3, 4)
    boot_preds.append(model.predict(noise)[0])
boot_counts = pd.Series(boot_preds).value_counts(normalize=True)
top_boot    = CLASS_LABELS.get(boot_counts.index[0], boot_counts.index[0])
top_pct     = boot_counts.iloc[0]

# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.4], gap="large")

with col1:
    st.markdown("### Predicted BMI Class")
    st.markdown(
        f"<div style='background:{color}18; border-left:5px solid {color}; "
        f"border-radius:10px; padding:1.2rem 1.4rem; margin-bottom:0.8rem;'>"
        f"<div style='font-size:1.8rem; font-weight:700; color:{color}; "
        f"letter-spacing:-0.02em;'>{label}</div>"
        f"<div style='font-size:0.9rem; color:#6b7280; margin-top:4px;'>"
        f"Model confidence: <b style='color:#111;'>{confidence*100:.1f}%</b></div>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.caption(
        f"Stability check (200 perturbations): **{top_boot}** predicted "
        f"{top_pct*100:.0f}% of the time. Uncertainty increases near class boundaries."
    )

    st.markdown("### Model Performance")
    st.markdown(
        f"<div class='metric-row'>"
        f"<div class='metric-card'>"
        f"<div class='metric-label'>5-Fold CV Accuracy</div>"
        f"<div class='metric-value'>{cv_mean*100:.1f}%</div>"
        f"<div class='metric-sub'>±{cv_ci*100:.1f}% &nbsp;(95% CI)</div>"
        f"</div>"
        f"<div class='metric-card'>"
        f"<div class='metric-label'>Baselines</div>"
        f"<div class='metric-value' style='font-size:0.95rem; padding-top:6px; line-height:1.7;'>"
        f"Random: 25.0%<br>Majority class: 35.4%</div>"
        f"</div></div>",
        unsafe_allow_html=True
    )
    st.caption(f"Model: {model_choice}")

with col2:
    st.markdown("### Predicted Class Probabilities")
    ordered_proba  = [proba_dict.get(c, 0) for c in CLASS_ORDER]
    ordered_labels = [CLASS_LABELS[c]       for c in CLASS_ORDER]
    bar_colors     = [CLASS_COLORS[c]       for c in CLASS_ORDER]
    alphas         = [0.95 if c == prediction else 0.4 for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(6, 3.4))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    for i, (lbl, val, col, alpha) in enumerate(
            zip(ordered_labels, ordered_proba, bar_colors, alphas)):
        ax.barh(i, val, color=col, alpha=alpha, height=0.55, linewidth=0)
        weight = 'bold' if CLASS_ORDER[i] == prediction else 'normal'
        ax.text(val + 0.012, i, f'{val*100:.1f}%',
                va='center', fontsize=10.5, fontweight=weight, color='#111827')

    ax.set_yticks(range(len(ordered_labels)))
    ax.set_yticklabels(ordered_labels, fontsize=10.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Probability', fontsize=9, color='#6b7280')
    ax.axvline(x=0.25, color='#9ca3af', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.253, len(CLASS_ORDER) - 0.3, 'Random baseline',
            ha='left', fontsize=7.5, color='#9ca3af')
    ax.tick_params(colors='#6b7280', labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color='#f3f4f6', linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=1.2)
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.divider()

with st.expander("About this model"):
    st.markdown("""
**Dataset:** NHANES (National Health and Nutrition Examination Survey), CDC -
https://www.cdc.gov/nchs/nhanes/index.htm

**Sample:** ~4,761 observations after deduplication | 80/20 stratified train/test split | `random_state=42`

| Model | 5-Fold CV Accuracy | 95% CI |
|---|---|---|
| Random Forest | ~46.0% | ±2.4% |
| Logistic Regression | ~45.1% | ±4.3% |
| Majority class baseline | 35.4% | - |
| Random baseline | 25.0% | - |

**Performance note:** This dataset does not include body weight. Since BMI is derived from
height and weight, predicting BMI category without weight imposes a ceiling on model accuracy.
Both models substantially outperform chance.

Feature importance reflects predictive associations only - not causal effects.
    """)

st.caption(
    "ECON 3916 Final Project  |  Cassandra Cinzori  |  "
    "Northeastern University  |  Spring 2026"
)
