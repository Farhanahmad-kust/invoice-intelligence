from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_error


def train_linear_regression(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def train_decision_tree(X_train, Y_train, max_depth=4):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, Y_train)
    return model

def train_random_forest(X_train, Y_train, max_depth=5, n_estimators=100):
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    model.fit(X_train, Y_train)
    return model


def evaluate_model(model, X_test, Y_test, model_name:str) -> dict:
    """
    Evaluate the performance of a trained model.

    Args:
        model: The trained model to evaluate.
        X_test: The test features.
        Y_test: The true target values for the test set.
        model_name (str): The name of the model being evaluated.

    Returns:
        dict: A dictionary containing the evaluation metrics (MAE, RMSE, R2).
    """

    predictions = model.predict(X_test)
    mae = mean_absolute_error(Y_test, predictions)
    rmse = mean_squared_error(Y_test, predictions)**0.5
    r2 = r2_score(Y_test, predictions)

    print(f"Model: {model_name} performance metrics:")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R2 Score: {r2}")


    return {
        "model_name": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
