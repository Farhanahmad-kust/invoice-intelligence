import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split

def load_vendor_invoice(db_path:str):
    """
    Load the vendor invoice data from the specified SQLite database.

    Args:
        db_path (str): The path to the SQLite database file.

    Returns:
        pd.DataFrame: The vendor invoice data.
    """

    import sqlite3
    import pandas as pd

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Load the vendor invoice data into a DataFrame
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    return df

def prepare_features(df: pd.DataFrame):
    """
    Prepare features and target variable for model training.

    Returns:
        tuple: A tuple containing the features (X) and target variable (y).
    """

    # Define the features and target variable
    # For example, let's assume we want to predict the freight cost based on the weight and distance
    features = ['Dollars','Quantity']
    target = 'Freight'

    X = df[features]
    y = df[target]

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Args:
        X (pd.DataFrame): The features.
        y (pd.Series): The target variable.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training and testing sets for features and target variable.
        """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
