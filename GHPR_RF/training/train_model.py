import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# Resolve project paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# =====================================================
# Load Dataset
# =====================================================

dataset_path = PROJECT_DIR / "dataset" / "defect_dataset.csv"

if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found:\n{dataset_path}")

df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")
print("Dataset Shape:", df.shape)

# =====================================================
# Features
# =====================================================

FEATURE_COLUMNS = [
    "cbo",
    "wmc",
    "totalMethods",
    "totalFields",
    "loc",
    "returnQty",
    "loopQty",
    "comparisonsQty",
    "tryCatchQty",
    "parenthesizedExpsQty",
    "stringLiteralsQty",
    "numbersQty",
    "assignmentsQty",
    "mathOperationsQty",
    "variablesQty",
    "maxNestedBlocks",
    "uniqueWordsQty"
]

TARGET_COLUMN = "defect"

# =====================================================
# Validate Dataset
# =====================================================

missing = [c for c in FEATURE_COLUMNS if c not in df.columns]

if TARGET_COLUMN not in df.columns:
    missing.append(TARGET_COLUMN)

if missing:
    raise ValueError(f"Missing columns:\n{missing}")

# =====================================================
# Prepare Data
# =====================================================

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# Random Forest Model
# =====================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_scaled,
    y_train
)

# =====================================================
# Evaluation
# =====================================================

y_pred = model.predict(X_test_scaled)

print("\n========================")
print("Random Forest Results")
print("========================")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# =====================================================
# Save Model
# =====================================================

models_dir = PROJECT_DIR / "models"
models_dir.mkdir(exist_ok=True)

joblib.dump(
    model,
    models_dir / "random_forest_model.pkl"
)

joblib.dump(
    scaler,
    models_dir / "scaler.pkl"
)

print("\nModel saved successfully!")