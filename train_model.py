"""AI4I 2020 — Machine Failure Prediction: train, compare, save best pipeline."""
import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             ConfusionMatrixDisplay)

RANDOM_STATE = 42

# ---------- 1. Load ----------
df = pd.read_csv("ai4i2020.csv")
display(df.head())                      # notebook: shows the data

# ---------- 2. Feature selection (prevents data leakage) ----------
# TWF/HDF/PWF/OSF/RNF are failure modes that DEFINE 'Machine failure' -> leakage, drop them.
FEATURES = ["Type", "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
TARGET = "Machine failure"
X, y = df[FEATURES], df[TARGET]

# ---------- 3. EDA ----------
print(X.describe().round(2))
print("Class balance:", y.value_counts(normalize=True).round(4).to_dict())  # ~3.4% failures
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(x=y, ax=ax[0]).set_title("Class balance")
sns.heatmap(pd.concat([X.select_dtypes("number"), y], axis=1).corr(),
            annot=True, fmt=".2f", ax=ax[1]).set_title("Correlations")
plt.tight_layout(); plt.show()

# ---------- 4. Split (stratified: keeps the rare failure class proportioned) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ---------- 5. Pipelines: identical preprocessing for every model ----------
ct = ColumnTransformer([
    ('num', StandardScaler(),[ 
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]),
    ('cat', OneHotEncoder(drop='first'),["Type"],)
])
model = Pipeline([
    ("ct", ct),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])
# ---------- 6. CLASSIFICATION ----------
model.fit(n_train,m_train)
m_pred = model.predict(n_test)
print(confusion_matrix(m_test, m_pred))
cr=classification_report(m_test, m_pred, zero_division=0)
print(cr)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
#----------------------7.DECISION TREE-------------------------
model.fit(n_train,m_train)
m_pred = model.predict(n_test)
print(confusion_matrix(m_test, m_pred))
cr=classification_report(m_test, m_pred, zero_division=0)
print(cr)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
#---------------------------MODEL SAVING------------------
os.makedirs(r"C:\Users\Aditya P. Shelar\Desktop\ML\models", exist_ok=True)
joblib.dump(dtc, r"C:\Users\Aditya P. Shelar\Desktop\ML\models\model.pkl")
print(os.path.exists(r"C:\Users\Aditya P. Shelar\Desktop\ML\models\model.pkl"))
#---------------------------MODEL PREDICTION-------------------------
m_pred=dtc.predict(n_test)
print(m_pred)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(m_test,m_pred)
print(cm)
from sklearn.metrics import classification_report
print(classification_report(m_test,m_pred,zero_division=0))
import joblib
model = joblib.load((r"C:\Users\Aditya P. Shelar\Desktop\ML\models\model.pkl"))
#---------------------------FINAL MODEL SAVE------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print(model.n_features_in_, list(model.feature_names_in_)) 