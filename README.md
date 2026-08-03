# Invoice Intelligence

A Streamlit MVP that estimates invoice freight and flags invoices that may need
additional review. The application performs inference only; it does not retrain
models or store uploaded invoice data.
<img width="1917" height="908" alt="image" src="https://github.com/user-attachments/assets/b46cfdd6-0f79-4f98-bb78-bd8cd9fb70ce" />



## Run locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. The app includes single-invoice
analysis, CSV batch processing, downloadable results, and model information.
<img width="1915" height="915" alt="image" src="https://github.com/user-attachments/assets/4bb826ae-ebb4-45c7-bf48-e916817fb5f8" />


## Batch CSV

Required columns:

- `invoice_quantity`
- `invoice_dollars`
- `total_item_quantity`
- `total_item_dollars`

`Freight` is optional. When it is missing or blank, the app uses predicted
freight. Extra source columns are preserved in the result.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Deploy to Streamlit Community Cloud

1. Push the application, `requirements.txt`, and both `.joblib` model files to
   GitHub. Do not commit `data/inventory.db`.
2. Create a Community Cloud app using `streamlit_app.py` as the entrypoint.
3. Select Python 3.14 in Advanced Settings.
4. Deploy. No secrets or external database are required.

This model provides decision support and must not be treated as a fraud or
payment-approval determination.
