import joblib
import pandas as pd
from pathlib import Path

# =====================================================
# Resolve Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# =====================================================
# Load Model
# =====================================================

model = joblib.load(
    PROJECT_DIR / "models" / "random_forest_model.pkl"
)

scaler = joblib.load(
    PROJECT_DIR / "models" / "scaler.pkl"
)

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

# =====================================================
# Prediction Function
# =====================================================

def predict_defect(metrics):

    missing = [
        feature for feature in FEATURE_COLUMNS
        if feature not in metrics
    ]

    if missing:
        raise ValueError(f"Missing metrics: {missing}")

    data = pd.DataFrame(
        [[metrics[col] for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    probability = model.predict_proba(data_scaled)

    return {
        "defect_prediction": int(prediction[0]),
        "defect_probability": round(float(probability[0][1]), 4)
    }


# =====================================================
# Example
# =====================================================

sample_metrics = {
    "cbo": 5,
    "wmc": 12,
    "totalMethods": 8,
    "totalFields": 4,
    "loc": 120,
    "returnQty": 3,
    "loopQty": 2,
    "comparisonsQty": 5,
    "tryCatchQty": 1,
    "parenthesizedExpsQty": 7,
    "stringLiteralsQty": 6,
    "numbersQty": 8,
    "assignmentsQty": 15,
    "mathOperationsQty": 10,
    "variablesQty": 12,
    "maxNestedBlocks": 4,
    "uniqueWordsQty": 90
}

if __name__ == "__main__":

    result = predict_defect(sample_metrics)

    print(result)