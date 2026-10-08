import os
import sys

import streamlit as st
import pandas as pd

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


from sdv.metadata import Metadata
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer
)

from src.evaluation.evaluation_pipeline import (
    evaluate_synthetic_data
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🧪",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🧪 Synthetic Data Generator")

st.write(
    "Generate synthetic tabular data and evaluate "
    "utility and basic privacy risks."
)

st.info(
    "This application is a prototype for research and demonstration. "
    "The privacy checks shown here are basic memorization checks, "
    "not formal privacy guarantees."
)


# --------------------------------------------------
# 1. Upload Dataset
# --------------------------------------------------

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

data = None


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success(
        f"Dataset loaded successfully: "
        f"{data.shape[0]} rows × {data.shape[1]} columns"
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head()
    )


    # --------------------------------------------------
    # 2. Select Model
    # --------------------------------------------------

    st.header("2. Select Synthetic Data Model")

    model_choice = st.selectbox(
        "Choose a generator",
        [
            "Gaussian Copula",
            "CTGAN",
            "CTGAN v2"
        ]
    )

    st.write(
        f"Selected model: **{model_choice}**"
    )


    # --------------------------------------------------
    # 3. Generate Synthetic Data
    # --------------------------------------------------

    st.header("3. Generate Synthetic Data")

    num_rows = st.number_input(
        "Number of synthetic rows",
        min_value=10,
        max_value=5000,
        value=100,
        step=10
    )

    generate_button = st.button(
        "Generate Synthetic Data",
        type="primary"
    )


    if generate_button:

        # ----------------------------------------------
        # Prepare model data
        # ----------------------------------------------

        model_data = data.copy()

        if "customer_id" in model_data.columns:

            model_data = model_data.drop(
                columns=["customer_id"]
            )


        # ----------------------------------------------
        # Detect metadata
        # ----------------------------------------------

        metadata = Metadata.detect_from_dataframe(
            data=model_data
        )

        if "city" in model_data.columns:

            metadata.update_column(
                column_name="city",
                sdtype="categorical"
            )


        # ----------------------------------------------
        # Train selected model
        # ----------------------------------------------

        if model_choice == "Gaussian Copula":

            with st.spinner(
                "Training Gaussian Copula and generating data..."
            ):

                synthesizer = GaussianCopulaSynthesizer(
                    metadata
                )

                synthesizer.fit(
                    model_data
                )

                synthetic_data = synthesizer.sample(
                    num_rows=int(num_rows)
                )


        elif model_choice == "CTGAN":

            with st.spinner(
                "Training CTGAN and generating data..."
            ):

                synthesizer = CTGANSynthesizer(
                    metadata,
                    epochs=300,
                    batch_size=100,
                    verbose=True
                )

                synthesizer.fit(
                    model_data
                )

                synthetic_data = synthesizer.sample(
                    num_rows=int(num_rows)
                )


        elif model_choice == "CTGAN v2":

            with st.spinner(
                "Training CTGAN v2 and generating data..."
            ):

                synthesizer = CTGANSynthesizer(
                    metadata,
                    epochs=300,
                    batch_size=100,
                    verbose=True
                )

                synthesizer.fit(
                    model_data
                )

                synthetic_data = synthesizer.sample(
                    num_rows=int(num_rows)
                )


        # ----------------------------------------------
        # Generation result
        # ----------------------------------------------

        st.success(
            f"Successfully generated "
            f"{len(synthetic_data)} synthetic rows."
        )

        st.subheader(
            "Synthetic Data Preview"
        )

        st.write(
            f"Generated dataset: "
            f"{len(synthetic_data)} rows × "
            f"{len(synthetic_data.columns)} columns"
        )

        st.dataframe(
            synthetic_data.head(10)
        )


        # ----------------------------------------------
        # 4. Evaluation
        # ----------------------------------------------

        st.header(
            "4. Utility & Privacy Evaluation"
        )

        numerical_columns = [
            column
            for column in [
                "age",
                "income",
                "credit_score",
                "purchase_amount"
            ]
            if column in model_data.columns
        ]


        with st.spinner(
            "Evaluating synthetic data..."
        ):

            evaluation_results = evaluate_synthetic_data(
                model_data,
                synthetic_data,
                numerical_columns
            )


        # ----------------------------------------------
        # Extract evaluation results
        # ----------------------------------------------

        correlation_similarity = (
            evaluation_results[
                "correlation"
            ][
                "correlation_similarity"
            ]
        )


        exact_match_rate = (
            evaluation_results[
                "exact_match"
            ][
                "exact_match_rate"
            ]
        )


        exact_matches = (
            evaluation_results[
                "exact_match"
            ][
                "exact_matches"
            ]
        )


        privacy_distance = (
            evaluation_results[
                "privacy_distance"
            ]
        )


        distance_ratio = (
            privacy_distance[
                "distance_ratio"
            ]
        )


        synthetic_avg_distance = (
            privacy_distance[
                "synthetic_average_nearest_distance"
            ]
        )


        training_avg_distance = (
            privacy_distance[
                "training_average_nearest_distance"
            ]
        )


        ks_results = (
            evaluation_results[
                "ks_metrics"
            ]
        )


        # ----------------------------------------------
        # Average KS statistic
        # ----------------------------------------------

        ks_statistics = [
            values["ks_statistic"]
            for values in ks_results.values()
        ]


        if len(ks_statistics) > 0:

            average_ks = sum(
                ks_statistics
            ) / len(ks_statistics)

        else:

            average_ks = 0.0


        # ----------------------------------------------
        # Basic evaluation rules
        # ----------------------------------------------

        utility_pass = (
            correlation_similarity >= 0.80
            and average_ks <= 0.15
        )


        privacy_pass = (
            exact_match_rate == 0
            and distance_ratio >= 1.0
        )


        overall_pass = (
            utility_pass
            and privacy_pass
        )


        # ----------------------------------------------
        # Evaluation Summary
        # ----------------------------------------------

        st.subheader(
            "📊 Evaluation Summary"
        )


        summary_col1, summary_col2, summary_col3 = (
            st.columns(3)
        )


        with summary_col1:

            if utility_pass:

                st.success(
                    "🟢 Utility\n\n"
                    "Basic utility checks passed."
                )

            else:

                st.warning(
                    "🟡 Utility\n\n"
                    "Utility needs improvement."
                )


        with summary_col2:

            if privacy_pass:

                st.success(
                    "🟢 Privacy\n\n"
                    "Basic privacy checks passed."
                )

            else:

                st.warning(
                    "🟡 Privacy\n\n"
                    "Privacy checks need review."
                )


        with summary_col3:

            if overall_pass:

                st.success(
                    "🟢 Overall\n\n"
                    "Synthetic data passed the basic checks."
                )

            else:

                st.warning(
                    "🟡 Overall\n\n"
                    "Synthetic data needs improvement."
                )


        st.caption(
            "These indicators are simple prototype thresholds "
            "for demonstration. They do not represent formal "
            "privacy guarantees or regulatory certification."
        )


        # ----------------------------------------------
        # Utility metrics
        # ----------------------------------------------

        st.subheader(
            "Utility: Correlation Similarity"
        )

        st.metric(
            "Correlation Similarity",
            f"{correlation_similarity:.4f}"
        )


        st.metric(
            "Average KS Statistic",
            f"{average_ks:.4f}"
        )


        # ----------------------------------------------
        # Privacy: Exact Match
        # ----------------------------------------------

        st.subheader(
            "Privacy: Exact Match Check"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Exact Match Rate",
                f"{exact_match_rate:.4f}"
            )


        with col2:

            st.metric(
                "Exact Training Records Found",
                exact_matches
            )


        # ----------------------------------------------
        # Privacy: Nearest Neighbor
        # ----------------------------------------------

        st.subheader(
            "Privacy: Nearest-Neighbor Distance"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Distance Ratio",
                f"{distance_ratio:.4f}"
            )


        with col2:

            st.metric(
                "Synthetic Avg. Distance",
                f"{synthetic_avg_distance:.4f}"
            )


        with col3:

            st.metric(
                "Training Avg. Distance",
                f"{training_avg_distance:.4f}"
            )


        # ----------------------------------------------
        # Distribution similarity
        # ----------------------------------------------

        st.subheader(
            "Utility: Distribution Similarity"
        )


        ks_table = pd.DataFrame(
            [
                {
                    "Column": column,
                    "KS Statistic": values[
                        "ks_statistic"
                    ],
                    "P-Value": values[
                        "p_value"
                    ]
                }
                for column, values
                in ks_results.items()
            ]
        )


        st.dataframe(
            ks_table,
            use_container_width=True
        )


        # ----------------------------------------------
        # 5. Download Synthetic Data
        # ----------------------------------------------

        st.subheader(
            "5. Download Synthetic Data"
        )


        csv_data = synthetic_data.to_csv(
            index=False
        )


        st.download_button(
            label="Download Synthetic Data",
            data=csv_data,
            file_name="synthetic_data.csv",
            mime="text/csv"
        )


    # --------------------------------------------------
        # 6. Model Comparison
        # --------------------------------------------------

        st.header(
            "6. Model Comparison"
        )


        comparison_file = os.path.join(
            project_root,
            "experiments",
            "results",
            "all_models_comparison.csv"
        )


        if os.path.exists(comparison_file):

            comparison_data = pd.read_csv(
                comparison_file
            )

            st.write(
                "Previously evaluated model results:"
            )

            st.dataframe(
                comparison_data,
                use_container_width=True
            )

        else:

            st.info(
                "Model comparison results are not available yet."
            )