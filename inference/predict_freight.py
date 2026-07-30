import joblib
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "freight_cost_prediction"
    / "models"
    / "best_linear_regression.joblib"
)


def load_model(model_path=MODEL_PATH):
    # Load the trained model from the specified path
    return joblib.load(model_path)


def predict_freight_cost(input_data, model=None):
    """Predict freight for one record using an optional preloaded model."""

    if model is None:
        model = load_model()

    # Validate and order features exactly as they were used during training.
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

    # Make predictions using the loaded model
    input_df["Predicted_Freight"] = model.predict(model_input).round(2)
    return input_df


if __name__ == "__main__":
    # Example input data for prediction
    example_input = {

        "Quantity": 100,
        "Dollars": 5000.0,

    }

    # Make a prediction
    prediction_result = predict_freight_cost(example_input)
    print("Prediction Result:")
    print(prediction_result)    

