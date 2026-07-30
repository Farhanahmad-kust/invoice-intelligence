"""Streamlit interface for the Invoice Intelligence MVP."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from inference.invoice_intelligence import (
    REQUIRED_INPUT_COLUMNS,
    analyze_invoice,
    analyze_invoice_batch,
)
from inference.predict_freight import load_model as load_freight_model
from inference.predict_invoice_flag import load_model as load_flagging_model


st.set_page_config(
    page_title="Invoice Intelligence",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #64748b;
        --surface: #f8fafc;
        --brand: #0f766e;
        --brand-soft: #ccfbf1;
        --danger: #b42318;
    }
    .stApp {
        background:
            radial-gradient(circle at top right, #e6fffb 0, transparent 32rem),
            #ffffff;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid #dbe7e5;
        border-radius: 18px;
        background: linear-gradient(135deg, #f0fdfa, #ffffff);
        margin-bottom: 1.4rem;
    }
    .hero-kicker {
        color: var(--brand);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.05;
        margin: 0.35rem 0 0.65rem;
    }
    .hero p {
        color: var(--muted);
        font-size: 1.05rem;
        margin: 0;
        max-width: 760px;
    }
    .status-card {
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        border: 1px solid #dbe7e5;
        background: white;
        min-height: 126px;
    }
    .status-card.flagged {
        border-color: #fecaca;
        background: #fff7f7;
    }
    .status-card.normal {
        border-color: #a7f3d0;
        background: #f0fdf4;
    }
    .status-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .status-value {
        color: var(--ink);
        font-size: 1.65rem;
        font-weight: 750;
        margin-top: 0.35rem;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.87rem;
    }
    div[data-testid="stForm"] {
        border-color: #dbe7e5;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.82);
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        background: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_models():
    """Load model artifacts once for the Streamlit process."""

    return load_freight_model(), load_flagging_model()


def render_header() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">Decision support for invoice review</div>
            <h1>Invoice Intelligence</h1>
            <p>
                Estimate freight, identify invoices that may need attention,
                and process structured invoice batches in one private session.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_single_invoice(freight_model, flagging_model) -> None:
    st.subheader("Analyze one invoice")
    st.caption(
        "Enter invoice-level totals and the matching sum of purchased line items."
    )

    freight_choice = st.radio(
        "Freight source",
        ("Use predicted freight", "Use actual freight"),
        horizontal=True,
        help=(
            "Use actual freight when it appears on the invoice. Otherwise, "
            "the freight model estimates it from invoice quantity and dollars."
        ),
    )

    with st.form("single_invoice_form"):
        invoice_column, item_column = st.columns(2)
        with invoice_column:
            st.markdown("#### Invoice totals")
            invoice_quantity = st.number_input(
                "Invoice quantity",
                min_value=0.01,
                value=100.0,
                step=1.0,
                help="Total quantity stated on the vendor invoice.",
            )
            invoice_dollars = st.number_input(
                "Invoice dollars",
                min_value=0.0,
                value=5000.0,
                step=100.0,
                format="%.2f",
                help="Total merchandise value stated on the invoice.",
            )
            actual_freight = st.number_input(
                "Actual freight",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.2f",
                disabled=freight_choice == "Use predicted freight",
                help="Used by the flagging model only when actual freight is selected.",
            )

        with item_column:
            st.markdown("#### Aggregated line items")
            total_item_quantity = st.number_input(
                "Total item quantity",
                min_value=0.01,
                value=100.0,
                step=1.0,
                help="Sum of quantities across all purchase lines.",
            )
            total_item_dollars = st.number_input(
                "Total item dollars",
                min_value=0.0,
                value=5000.0,
                step=100.0,
                format="%.2f",
                help="Sum of dollar values across all purchase lines.",
            )
            st.info(
                "Large differences between invoice and line-item totals can "
                "increase the review signal.",
                icon="ℹ️",
            )

        submitted = st.form_submit_button(
            "Analyze invoice",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        payload = {
            "invoice_quantity": invoice_quantity,
            "invoice_dollars": invoice_dollars,
            "total_item_quantity": total_item_quantity,
            "total_item_dollars": total_item_dollars,
        }
        if freight_choice == "Use actual freight":
            payload["Freight"] = actual_freight

        try:
            with st.spinner("Running invoice intelligence models..."):
                st.session_state["single_result"] = analyze_invoice(
                    payload,
                    freight_model=freight_model,
                    flagging_model=flagging_model,
                )
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.pop("single_result", None)

    result = st.session_state.get("single_result")
    if not result:
        return

    st.markdown("### Analysis result")
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric(
        "Predicted freight",
        f"${result['Predicted_Freight']:,.2f}",
    )
    metric_two.metric(
        "Freight used",
        f"${result['Freight_Used']:,.2f}",
    )
    metric_three.metric(
        "Flagged probability",
        f"{result['Flagged_Probability']:.1%}",
    )

    status_class = "flagged" if result["Flag_Invoice"] else "normal"
    status_message = (
        "Send this invoice for additional review."
        if result["Flag_Invoice"]
        else "No review flag was raised by the model."
    )
    st.markdown(
        f"""
        <div class="status-card {status_class}">
            <div class="status-label">Invoice review status</div>
            <div class="status-value">{result["Risk_Label"]}</div>
            <div class="small-note">{status_message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _template_csv() -> bytes:
    template = pd.DataFrame(
        [
            {
                "invoice_quantity": 100,
                "invoice_dollars": 5000.00,
                "total_item_quantity": 100,
                "total_item_dollars": 5000.00,
                "Freight": "",
            }
        ]
    )
    return template.to_csv(index=False).encode("utf-8")


def render_batch_processing(freight_model, flagging_model) -> None:
    st.subheader("Process an invoice batch")
    st.caption(
        "Upload a CSV. Leave Freight blank to use the predicted freight value."
    )

    action_column, detail_column = st.columns([1, 2])
    with action_column:
        st.download_button(
            "Download CSV template",
            data=_template_csv(),
            file_name="invoice_intelligence_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with detail_column:
        st.markdown(
            "**Required columns:** "
            + ", ".join(f"`{column}`" for column in REQUIRED_INPUT_COLUMNS)
        )

    uploaded_file = st.file_uploader(
        "Invoice CSV",
        type=("csv",),
        help="The original columns are preserved in the downloaded result.",
    )
    if uploaded_file is None:
        st.info("Upload a CSV to preview and analyze invoice rows.", icon="📄")
        return

    try:
        raw_bytes = uploaded_file.getvalue()
        invoices = pd.read_csv(io.BytesIO(raw_bytes))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the CSV: {exc}")
        return

    if invoices.empty:
        st.error("The uploaded CSV contains no invoice rows.")
        return

    st.markdown("#### Upload preview")
    summary_one, summary_two, summary_three = st.columns(3)
    summary_one.metric("Rows", f"{len(invoices):,}")
    summary_two.metric("Columns", len(invoices.columns))
    summary_three.metric(
        "Actual freight values",
        int(invoices.get("Freight", pd.Series(dtype=float)).notna().sum()),
    )
    st.dataframe(invoices.head(20), use_container_width=True, hide_index=True)

    if st.button(
        "Analyze batch",
        type="primary",
        use_container_width=True,
    ):
        progress = st.progress(10, text="Validating and loading invoice rows...")
        try:
            progress.progress(45, text="Running freight predictions...")
            batch_result = analyze_invoice_batch(
                invoices,
                freight_model=freight_model,
                flagging_model=flagging_model,
            )
            progress.progress(100, text="Analysis complete")
            st.session_state["batch_result"] = batch_result
        except ValueError as exc:
            progress.empty()
            st.error(str(exc))
            st.session_state.pop("batch_result", None)

    batch_result = st.session_state.get("batch_result")
    if batch_result is None:
        return

    error_count = int(batch_result["Error"].ne("").sum())
    successful = len(batch_result) - error_count
    flagged = int(
        pd.to_numeric(
            batch_result["Flag_Invoice"],
            errors="coerce",
        ).fillna(0).sum()
    )

    st.markdown("### Batch results")
    result_one, result_two, result_three = st.columns(3)
    result_one.metric("Successful rows", f"{successful:,}")
    result_two.metric("Flagged invoices", f"{flagged:,}")
    result_three.metric("Rows with errors", f"{error_count:,}")

    if error_count:
        st.warning(
            "Rows with errors were preserved. Review the Error column before use."
        )

    st.dataframe(batch_result, use_container_width=True, hide_index=True)
    st.download_button(
        "Download analyzed CSV",
        data=batch_result.to_csv(index=False).encode("utf-8"),
        file_name="invoice_intelligence_results.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )


def render_model_information() -> None:
    st.subheader("How the models work")

    freight_tab, flagging_tab, limitations_tab = st.tabs(
        ("Freight model", "Flagging model", "Limitations")
    )
    with freight_tab:
        st.markdown(
            """
            The freight estimator is a linear regression model using:

            - Invoice dollars
            - Invoice quantity

            Its test performance was approximately **MAE 24.46** and
            **R² 0.97**. Predictions are estimates in the same currency unit
            used during training.
            """
        )

    with flagging_tab:
        st.markdown(
            """
            The invoice review model is a random forest classifier using:

            - Invoice quantity and dollars
            - Freight
            - Aggregated line-item quantity and dollars

            The current evaluation showed approximately **0.93 flagged-class
            precision**, **0.69 flagged-class recall**, and **0.79 F1**.
            A probability of 0.50 is the default classification threshold.
            """
        )

    with limitations_tab:
        st.warning(
            "This application supports review decisions; it does not determine "
            "fraud, payment approval, or vendor misconduct.",
            icon="⚠️",
        )
        st.markdown(
            """
            - Flagging labels were generated from review rules rather than
              independently confirmed fraud outcomes.
            - A recall of 0.69 means some invoices meeting the learned review
              pattern will not be flagged.
            - Predictions should be combined with source-document checks and
              organizational controls.
            - Uploaded data and results stay in the active session and are not
              written to a database by this application.
            """
        )


def main() -> None:
    render_header()

    st.sidebar.markdown("### Invoice Intelligence")
    view = st.sidebar.radio(
        "Workspace",
        ("Single Invoice", "Batch Processing", "Model Information"),
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "No accounts · No stored history · Decision support only"
    )

    try:
        freight_model, flagging_model = get_models()
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Model artifacts could not be loaded: {exc}")
        st.stop()

    if view == "Single Invoice":
        render_single_invoice(freight_model, flagging_model)
    elif view == "Batch Processing":
        render_batch_processing(freight_model, flagging_model)
    else:
        render_model_information()


if __name__ == "__main__":
    main()
