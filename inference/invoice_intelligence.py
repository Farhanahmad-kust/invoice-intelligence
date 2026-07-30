"""Integrated, validated inference for single invoices and CSV batches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from inference.predict_freight import load_model as load_freight_model
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import load_model as load_flagging_model
from inference.predict_invoice_flag import predict_invoice_flag


REQUIRED_INPUT_COLUMNS = (
    "invoice_quantity",
    "invoice_dollars",
    "total_item_quantity",
    "total_item_dollars",
)
OPTIONAL_INPUT_COLUMNS = ("Freight",)
OUTPUT_COLUMNS = (
    "Predicted_Freight",
    "Freight_Used",
    "Flag_Invoice",
    "Risk_Label",
    "Flagged_Probability",
    "Error",
)


def _validated_number(
    value: Any,
    field: str,
    *,
    strictly_positive: bool,
) -> float:
    if value is None or pd.isna(value):
        raise ValueError(f"{field} is required")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc

    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if strictly_positive and number <= 0:
        raise ValueError(f"{field} must be greater than 0")
    if not strictly_positive and number < 0:
        raise ValueError(f"{field} cannot be negative")
    return number


def validate_invoice_input(input_data: Mapping[str, Any]) -> dict[str, float]:
    """Validate and normalize the integrated invoice input contract."""

    missing = set(REQUIRED_INPUT_COLUMNS) - set(input_data)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    normalized = {
        "invoice_quantity": _validated_number(
            input_data["invoice_quantity"],
            "invoice_quantity",
            strictly_positive=True,
        ),
        "invoice_dollars": _validated_number(
            input_data["invoice_dollars"],
            "invoice_dollars",
            strictly_positive=False,
        ),
        "total_item_quantity": _validated_number(
            input_data["total_item_quantity"],
            "total_item_quantity",
            strictly_positive=True,
        ),
        "total_item_dollars": _validated_number(
            input_data["total_item_dollars"],
            "total_item_dollars",
            strictly_positive=False,
        ),
    }

    freight = input_data.get("Freight")
    if freight is not None and not pd.isna(freight) and freight != "":
        normalized["Freight"] = _validated_number(
            freight,
            "Freight",
            strictly_positive=False,
        )

    return normalized


def analyze_invoice(
    input_data: Mapping[str, Any],
    *,
    freight_model=None,
    flagging_model=None,
) -> dict[str, Any]:
    """Run freight prediction and invoice flagging for one invoice."""

    normalized = validate_invoice_input(input_data)
    if freight_model is None:
        freight_model = load_freight_model()
    if flagging_model is None:
        flagging_model = load_flagging_model()

    freight_input = {
        "Quantity": normalized["invoice_quantity"],
        "Dollars": normalized["invoice_dollars"],
    }
    freight_result = predict_freight_cost(
        freight_input,
        model=freight_model,
    )
    predicted_freight = float(
        freight_result["Predicted_Freight"].iloc[0]
    )
    freight_used = normalized.get("Freight", predicted_freight)

    flagging_input = {
        "invoice_quantity": normalized["invoice_quantity"],
        "invoice_dollars": normalized["invoice_dollars"],
        "Freight": freight_used,
        "total_item_quantity": normalized["total_item_quantity"],
        "total_item_dollars": normalized["total_item_dollars"],
    }
    flagging_result = predict_invoice_flag(
        flagging_input,
        model=flagging_model,
    )

    return {
        **normalized,
        "Predicted_Freight": predicted_freight,
        "Freight_Used": float(freight_used),
        "Flag_Invoice": int(flagging_result["Flag_Invoice"].iloc[0]),
        "Risk_Label": str(flagging_result["Risk_Label"].iloc[0]),
        "Flagged_Probability": float(
            flagging_result["Flagged_Probability"].iloc[0]
        ),
        "Error": "",
    }


def analyze_invoice_batch(
    invoices: pd.DataFrame,
    *,
    freight_model=None,
    flagging_model=None,
) -> pd.DataFrame:
    """Process a CSV-shaped DataFrame while isolating row-level errors."""

    if invoices.empty:
        raise ValueError("The uploaded CSV contains no invoice rows")

    missing_columns = set(REQUIRED_INPUT_COLUMNS) - set(invoices.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {sorted(missing_columns)}"
        )

    if freight_model is None:
        freight_model = load_freight_model()
    if flagging_model is None:
        flagging_model = load_flagging_model()

    results = invoices.copy()
    for column in OUTPUT_COLUMNS:
        results[column] = pd.NA

    for index, row in invoices.iterrows():
        try:
            prediction = analyze_invoice(
                row.to_dict(),
                freight_model=freight_model,
                flagging_model=flagging_model,
            )
            for column in OUTPUT_COLUMNS:
                results.at[index, column] = prediction[column]
        except (TypeError, ValueError) as exc:
            results.at[index, "Error"] = str(exc)

    results["Error"] = results["Error"].fillna("")
    return results
