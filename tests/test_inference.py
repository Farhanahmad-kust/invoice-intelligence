import unittest
from pathlib import Path

import pandas as pd

from inference.invoice_intelligence import (
    OUTPUT_COLUMNS,
    analyze_invoice,
    analyze_invoice_batch,
)
from inference.predict_freight import load_model as load_freight_model
from inference.predict_invoice_flag import load_model as load_flagging_model


class InvoiceIntelligenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.freight_model = load_freight_model()
        cls.flagging_model = load_flagging_model()
        cls.base_invoice = {
            "invoice_quantity": 100,
            "invoice_dollars": 5000.0,
            "total_item_quantity": 100,
            "total_item_dollars": 5000.0,
        }

    def analyze(self, payload):
        return analyze_invoice(
            payload,
            freight_model=self.freight_model,
            flagging_model=self.flagging_model,
        )

    def test_model_feature_contracts(self):
        self.assertEqual(
            list(self.freight_model.feature_names_in_),
            ["Dollars", "Quantity"],
        )
        self.assertEqual(
            list(self.flagging_model.feature_names_in_),
            [
                "invoice_quantity",
                "invoice_dollars",
                "Freight",
                "total_item_quantity",
                "total_item_dollars",
            ],
        )

    def test_single_prediction_uses_predicted_freight_when_missing(self):
        result = self.analyze(self.base_invoice)
        self.assertAlmostEqual(
            result["Freight_Used"],
            result["Predicted_Freight"],
        )
        self.assertIn(result["Risk_Label"], {"Normal", "Flagged"})
        self.assertGreaterEqual(result["Flagged_Probability"], 0)
        self.assertLessEqual(result["Flagged_Probability"], 1)

    def test_single_prediction_uses_actual_freight(self):
        payload = {**self.base_invoice, "Freight": 42.75}
        result = self.analyze(payload)
        self.assertEqual(result["Freight_Used"], 42.75)

    def test_batch_accepts_reordered_and_extra_columns(self):
        invoices = pd.DataFrame(
            [
                {
                    "reference": "INV-1",
                    "total_item_dollars": 5000,
                    "invoice_dollars": 5000,
                    "total_item_quantity": 100,
                    "invoice_quantity": 100,
                }
            ]
        )
        result = analyze_invoice_batch(
            invoices,
            freight_model=self.freight_model,
            flagging_model=self.flagging_model,
        )
        self.assertEqual(result.loc[0, "reference"], "INV-1")
        self.assertEqual(result.loc[0, "Error"], "")
        for column in OUTPUT_COLUMNS:
            self.assertIn(column, result.columns)

    def test_batch_isolates_invalid_rows(self):
        invoices = pd.DataFrame(
            [
                self.base_invoice,
                {**self.base_invoice, "invoice_quantity": "invalid"},
            ]
        )
        result = analyze_invoice_batch(
            invoices,
            freight_model=self.freight_model,
            flagging_model=self.flagging_model,
        )
        self.assertEqual(result.loc[0, "Error"], "")
        self.assertIn("must be numeric", result.loc[1, "Error"])

    def test_batch_rejects_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "Missing required CSV columns"):
            analyze_invoice_batch(
                pd.DataFrame([{"invoice_quantity": 1}]),
                freight_model=self.freight_model,
                flagging_model=self.flagging_model,
            )

    def test_batch_rejects_empty_csv(self):
        with self.assertRaisesRegex(ValueError, "contains no invoice rows"):
            analyze_invoice_batch(
                pd.DataFrame(),
                freight_model=self.freight_model,
                flagging_model=self.flagging_model,
            )

    def test_inference_does_not_modify_sqlite_database(self):
        database_path = Path("data/inventory.db")
        before = database_path.stat().st_mtime_ns
        self.analyze(self.base_invoice)
        after = database_path.stat().st_mtime_ns
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
