from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "invoice_flagging"
    / "models"
    / "invoice_flagging_model.joblib"
)


def load_model(model_path=MODEL_PATH):
    """Load the trained invoice-flagging classifier."""

    return joblib.load(model_path)


def predict_invoice_flag(input_data, model=None):
    """Predict whether one invoice should be flagged for review."""

    if model is None:
        model = load_model()
    required_features = list(model.feature_names_in_)

    missing_features = set(required_features) - set(input_data)
    unexpected_features = set(input_data) - set(required_features)

    if missing_features:
        raise ValueError(
            f"Missing required features: {sorted(missing_features)}"
        )
    if unexpected_features:
        raise ValueError(
            f"Unexpected features: {sorted(unexpected_features)}"
        )

    input_df = pd.DataFrame([input_data])
    model_input = input_df[required_features]

    prediction = int(model.predict(model_input)[0])
    flagged_class_index = list(model.classes_).index(1)
    flagged_probability = float(
        model.predict_proba(model_input)[0, flagged_class_index]
    )

    input_df["Flag_Invoice"] = prediction
    input_df["Risk_Label"] = "Flagged" if prediction == 1 else "Normal"
    input_df["Flagged_Probability"] = round(flagged_probability, 4)
    return input_df


if __name__ == "__main__":
    example_input = {
        "invoice_quantity": 100,
        "invoice_dollars": 5000.0,
        "Freight": 31.48,
        "total_item_quantity": 80,
        "total_item_dollars": 5000.0,
    }

    prediction_result = predict_invoice_flag(example_input)
    print("Prediction Result:")
    print(prediction_result.to_string(index=False))
