import joblib
import pandas as pd
from pathlib import Path

# =====================================================
# Resolve Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# =====================================================
# Load Model & Scaler
# =====================================================

model = joblib.load(
    PROJECT_DIR / "models" / "logistic_model.pkl"
)

scaler = joblib.load(
    PROJECT_DIR / "models" / "scaler.pkl"
)

# =====================================================
# Feature Order
# MUST be identical to the training script
# =====================================================

FEATURE_COLUMNS = [
    "LOC",
    "Cyclomatic_Complexity",
    "Function_Count",
    "Class_Count",
    "Method_Count",
    "Variable_Count",
    "Loop_Count",
    "Condition_Count",
    "Return_Count",
    "Try_Catch_Count",
    "String_Literal_Count",
    "Number_Literal_Count",
    "Math_Operation_Count",
    "Max_Nesting_Depth"
]

# =====================================================
# Prediction Function
# =====================================================

def predict_defect(metrics: dict):

    # Check for missing features
    missing = [feature for feature in FEATURE_COLUMNS if feature not in metrics]

    if missing:
        raise ValueError(
            f"Missing required metrics: {missing}"
        )

    # Create DataFrame in the correct column order
    data = pd.DataFrame(
        [[metrics[col] for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    # Scale input
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)

    # Probability
    probability = model.predict_proba(data_scaled)

    return {
        "defect_prediction": int(prediction[0]),
        "defect_probability": round(float(probability[0][1]), 4)
    }


# =====================================================
# Example Input
# =====================================================

sample_metrics = {

    "LOC": 120,
    "Cyclomatic_Complexity": 12,
    "Function_Count": 6,
    "Class_Count": 2,
    "Method_Count": 6,
    "Variable_Count": 18,
    "Loop_Count": 3,
    "Condition_Count": 5,
    "Return_Count": 4,
    "Try_Catch_Count": 1,
    "String_Literal_Count": 8,
    "Number_Literal_Count": 10,
    "Math_Operation_Count": 15,
    "Max_Nesting_Depth": 4

}

# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    result = predict_defect(sample_metrics)

    print("\nPrediction Result")
    print(result)