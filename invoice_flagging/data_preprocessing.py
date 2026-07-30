import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "inventory.db"


def load_invoice_data():
    # Load invoice data from the SQLite database
    query = """
    WITH purchase_agg AS (
        SELECT
            p.PONumber,
            COUNT(DISTINCT p.Brand) AS total_brands,
            SUM(p.Quantity) AS total_item_quantity,
            SUM(p.Dollars) AS total_item_dollars,
            AVG(
                julianday(p.ReceivingDate) - julianday(p.PODate)
            ) AS avg_receiving_delay
        FROM purchases AS p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Quantity AS invoice_quantity,
        vi.Dollars AS invoice_dollars,
        vi.Freight,
        julianday(vi.InvoiceDate) - julianday(vi.PODate)
            AS days_po_to_invoice,
        julianday(vi.PayDate) - julianday(vi.InvoiceDate)
            AS days_to_pay,
        pa.total_brands,
        pa.total_item_quantity,
        pa.total_item_dollars,
        pa.avg_receiving_delay
    FROM vendor_invoice AS vi
    LEFT JOIN purchase_agg AS pa
        ON vi.PONumber = pa.PONumber
    """

    with sqlite3.connect(DATABASE_PATH) as connection:
        return pd.read_sql_query(query, connection)

def create_invoice_risk_label(row):
    # Create a risk label based on the invoice data
    quantity_difference = abs(
        row["invoice_quantity"] - row["total_item_quantity"]
    )
    if quantity_difference > 5:
        return 1  # High risk
    if row["avg_receiving_delay"] > 10:
        return 1  # High risk
    return 0  # Low risk

def apply_labels(df):
    # Apply the risk label creation function to the DataFrame
    df['flag_invoice'] = df.apply(create_invoice_risk_label, axis=1)
    return df

def split_data(df, features, target):
    # Split the data into training and testing sets
    X = df[features]
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

def scale_features(X_train, X_test, scaler_path):
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled


