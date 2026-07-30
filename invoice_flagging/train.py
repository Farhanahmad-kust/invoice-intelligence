from pathlib import Path

import joblib

from .data_preprocessing import (
    apply_labels,
    load_invoice_data,
    split_data,
)
from .model_evaluation import (
    evaluate_classifier,
    train_random_forest,
)

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

TARGET = "flag_invoice"


def main():
    package_dir = Path(__file__).resolve().parent
    model_dir = package_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = apply_labels(load_invoice_data())

    X_train, X_test, y_train, y_test = split_data(
        df,
        FEATURES,
        TARGET,
    )

    model = train_random_forest(X_train, y_train)

    evaluate_classifier(
        model,
        X_test,
        y_test,
        "Random Forest",
    )

    model_path = model_dir / "invoice_flagging_model.joblib"
    joblib.dump(model, model_path)

    print("Using notebook-selected parameters:", model.get_params())
    print("Model saved to:", model_path)


if __name__ == "__main__":
    main()
