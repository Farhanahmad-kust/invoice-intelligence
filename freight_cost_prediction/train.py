from pathlib import Path

import joblib

if __package__:
    from .data_preprocessing import (
        load_vendor_invoice,
        prepare_features,
        split_data,
    )
    from .model_evaluation import (
        evaluate_model,
        train_decision_tree,
        train_linear_regression,
        train_random_forest,
    )
else:
    from data_preprocessing import (
        load_vendor_invoice,
        prepare_features,
        split_data,
    )
    from model_evaluation import (
        evaluate_model,
        train_decision_tree,
        train_linear_regression,
        train_random_forest,
    )

def main():
    # Load the vendor invoice data
    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "data" / "inventory.db"
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)

    df = load_vendor_invoice(db_path)

    X, y = prepare_features(df)
    X_train, X_test, Y_train, Y_test = split_data(X, y)


#train models 
    lr_model = train_linear_regression(X_train, Y_train)
    dt_model = train_decision_tree(X_train, Y_train)
    rf_model = train_random_forest(X_train, Y_train)

    # Evaluate models
    evaluated_models = [
        (
            lr_model,
            evaluate_model(
                lr_model, X_test, Y_test, "linear_regression"
            ),
        ),
        (
            dt_model,
            evaluate_model(
                dt_model, X_test, Y_test, "decision_tree"
            ),
        ),
        (
            rf_model,
            evaluate_model(
                rf_model, X_test, Y_test, "random_forest"
            ),
        ),
    ]

    #select best model based on lowest mae
    best_model, best_model_info = min(
        evaluated_models,
        key=lambda item: item[1]["MAE"],
    )
    best_model_name = best_model_info["model_name"]


    # Save the best model
    model_path = model_dir / f"best_{best_model_name}.joblib"
    joblib.dump(best_model, model_path)


    print(f"Best model '{best_model_name}' saved to {model_path}")


if __name__ == "__main__":
    main()
