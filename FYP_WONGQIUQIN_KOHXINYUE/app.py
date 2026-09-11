# Loan Payback Prediction System

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from function.loader import load_resources
from function.preprocessing import preprocess_input

from function.prediction import (
    predict_model,
    get_model_display_name
)

from function.rfe import (
    get_rfe_summary,
    create_rfe_feature_table,
    prepare_rfe_input,
    create_rfe_summary_table,
    create_rfe_auc_chart_data,
    create_rfe_model_comparison_table,
    get_rfe_interpretation
)

from function.pca import (
    get_pca_summary,
    create_pca_component_table,
    create_pca_summary_table,
    create_formatted_variance_table,
    create_pca_variance_chart_data,
    create_pca_loading_summary,
    create_pca_model_comparison_by_model,
    get_pca_interpretation
)

from function.shap import (
    calculate_shap_explanation,
    create_shap_table,
    create_shap_waterfall_plot
)

from function.lime import (
    calculate_lime_explanation,
    create_lime_table,
    create_lime_bar_plot
)

from function.helper import (
    safe_index,
    clean_text,
    get_sorted_options,
    display_actual_label,
    display_prediction_result,
    display_input_dataframe,
    validate_probabilities,
    display_section_error
)



st.set_page_config(
    page_title="Loan Payback Prediction",
    layout="wide"
)

@st.cache_resource
def get_resources():
    """
    Load and cache models, transformers and supporting files.
    """

    return load_resources()

try:
    resources = get_resources()

except Exception as error:
    st.error(
        "The application resources could not be loaded."
    )

    st.exception(
        error
    )

    st.stop()

train_data = resources[
    "train_data"
]

model_columns = list(
    resources["model_columns"]
)

st.sidebar.header(
    "Model Selection"
)

input_representation = st.sidebar.selectbox(
    "Input Representation",
    [
        "Original",
        "RFE",
        "PCA"
    ]
)

prediction_type = input_representation

selected_model = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "GAM",
        "EBM",
        "BLR"
    ]
)

model_display_name = get_model_display_name(
    prediction_type,
    selected_model
)

if prediction_type == "Original":
    selected_model_object = resources[
        "models"
    ].get(
        selected_model
    )

elif prediction_type == "RFE":
    selected_model_object = resources[
        "rfe_models"
    ].get(
        selected_model
    )

else:
    selected_model_object = resources[
        "pca_models"
    ].get(
        selected_model
    )

if selected_model_object is None:
    st.sidebar.error(
        f"{model_display_name} could not be loaded."
    )

else:
    st.sidebar.success(
        f"{model_display_name} loaded successfully."
    )

st.sidebar.markdown("---")

threshold = st.sidebar.slider(
    "Default Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

st.sidebar.caption(
    "The borrower is classified as default when the "
    "default probability is equal to or higher than "
    "the selected threshold."
)

st.title(
    "Loan Payback Prediction System"
)

st.write(
    "Enter borrower details manually or randomly select "
    "a borrower from the training dataset."
)

st.header(
    "1. Sample Selection"
)

sample_choice = st.selectbox(
    "Choose Input Type",
    [
        "Manual Input",
        "Upload CSV File",
        "Random Paid Back Case (1.0)",
        "Random Default Case (0.0)"
    ]
)

required_input_columns = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade"
]

sample_row = None
uploaded_data = None

if sample_choice == "Upload CSV File":
    uploaded_file = st.file_uploader(
        "Upload Borrower CSV File",
        type=["csv"],
        help=(
            "The CSV must contain the required borrower "
            "information columns."
        )
    )

    if uploaded_file is None:
        st.info(
            "Upload a CSV file to load borrower information."
        )

        st.stop()

    try:
        uploaded_data = pd.read_csv(
            uploaded_file
        )

    except Exception as error:
        st.error(
            "The uploaded CSV file could not be read."
        )

        st.exception(
            error
        )

        st.stop()


    if uploaded_data.empty:
        st.error(
            "The uploaded CSV file does not contain any rows."
        )

        st.stop()


    missing_columns = [
        column
        for column in required_input_columns
        if column not in uploaded_data.columns
    ]


    if missing_columns:
        st.error(
            "The uploaded CSV file is missing the following "
            "required columns:"
        )

        st.write(
            missing_columns
        )

        st.stop()

    st.success(
        f"CSV file loaded successfully. "
        f"{len(uploaded_data):,} borrower record(s) found."
    )


    with st.expander(
        "View Uploaded CSV Data",
        expanded=False
    ):
        st.dataframe(
            uploaded_data,
            width="stretch",
            hide_index=True
        )

    if len(uploaded_data) > 1:
        uploaded_row_number = st.selectbox(
            "Choose Borrower Row",
            options=list(
                range(len(uploaded_data))
            ),
            format_func=lambda row_index: (
                f"Row {row_index + 1}"
            )
        )

    else:
        uploaded_row_number = 0


    sample_row = uploaded_data.iloc[
        uploaded_row_number
    ].copy()

elif sample_choice == "Random Paid Back Case (1.0)":
    paid_back_rows = train_data[
        train_data["loan_paid_back"] == 1
    ]

    if len(paid_back_rows) == 0:
        st.error(
            "No paid-back records were found in train.csv."
        )

        st.stop()

    sample_state_key = "random_paid_back_row_index"
    sample_revision_key = "random_paid_back_revision"

    if sample_state_key not in st.session_state:
        st.session_state[sample_state_key] = (
            paid_back_rows.sample(n=1).index[0]
        )
        st.session_state[sample_revision_key] = 0

    if st.button("Choose Another Random Paid Back Case"):
        st.session_state[sample_state_key] = (
            paid_back_rows.sample(n=1).index[0]
        )
        st.session_state[sample_revision_key] += 1

    sample_row = train_data.loc[
        st.session_state[sample_state_key]
    ]

elif sample_choice == "Random Default Case (0.0)":
    default_rows = train_data[
        train_data["loan_paid_back"] == 0
    ]

    if len(default_rows) == 0:
        st.error(
            "No default records were found in train.csv."
        )

        st.stop()

    sample_state_key = "random_default_row_index"
    sample_revision_key = "random_default_revision"

    if sample_state_key not in st.session_state:
        st.session_state[sample_state_key] = (
            default_rows.sample(n=1).index[0]
        )
        st.session_state[sample_revision_key] = 0

    if st.button("Choose Another Random Default Case"):
        st.session_state[sample_state_key] = (
            default_rows.sample(n=1).index[0]
        )
        st.session_state[sample_revision_key] += 1

    sample_row = train_data.loc[
        st.session_state[sample_state_key]
    ]

else:
    sample_row = None

if sample_choice == "Random Paid Back Case (1.0)":
    input_widget_key = (
        f"paid_back_{st.session_state['random_paid_back_revision']}"
    )
elif sample_choice == "Random Default Case (0.0)":
    input_widget_key = (
        f"default_{st.session_state['random_default_revision']}"
    )
else:
    input_widget_key = sample_choice

st.header(
    "2. Borrower Information"
)

input_col1, input_col2 = st.columns(
    2
)

with input_col1:
    annual_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=(
            float(
                sample_row["annual_income"]
            )
            if sample_row is not None
            else 50000.0
        ),
        step=1000.0,
        key=f"annual_income_{input_widget_key}"
    )

    debt_to_income_ratio = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        value=(
            float(
                sample_row[
                    "debt_to_income_ratio"
                ]
            )
            if sample_row is not None
            else 0.30
        ),
        step=0.01,
        format="%.4f",
        key=f"debt_to_income_ratio_{input_widget_key}"
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=(
            int(
                sample_row["credit_score"]
            )
            if sample_row is not None
            else 700
        ),
        step=1,
        key=f"credit_score_{input_widget_key}"
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=(
            float(
                sample_row["loan_amount"]
            )
            if sample_row is not None
            else 15000.0
        ),
        step=500.0,
        key=f"loan_amount_{input_widget_key}"
    )

    interest_rate = st.number_input(
        "Interest Rate",
        min_value=0.0,
        value=(
            float(
                sample_row["interest_rate"]
            )
            if sample_row is not None
            else 5.5
        ),
        step=0.1,
        format="%.4f",
        key=f"interest_rate_{input_widget_key}"
    )

with input_col2:
    gender_options = get_sorted_options(
        train_data,
        "gender"
    )

    gender_value = (
        clean_text(
            sample_row["gender"]
        )
        if sample_row is not None
        else gender_options[0]
    )

    gender = st.selectbox(
        "Gender",
        gender_options,
        index=safe_index(
            gender_options,
            gender_value
        ),
        key=f"gender_{input_widget_key}"
    )

    marital_options = get_sorted_options(
        train_data,
        "marital_status"
    )

    marital_value = (
        clean_text(
            sample_row["marital_status"]
        )
        if sample_row is not None
        else marital_options[0]
    )

    marital_status = st.selectbox(
        "Marital Status",
        marital_options,
        index=safe_index(
            marital_options,
            marital_value
        ),
        key=f"marital_status_{input_widget_key}"
    )

    education_options = get_sorted_options(
        train_data,
        "education_level"
    )

    education_value = (
        clean_text(
            sample_row["education_level"]
        )
        if sample_row is not None
        else education_options[0]
    )

    education_level = st.selectbox(
        "Education Level",
        education_options,
        index=safe_index(
            education_options,
            education_value
        ),
        key=f"education_level_{input_widget_key}"
    )

    employment_options = get_sorted_options(
        train_data,
        "employment_status"
    )

    employment_value = (
        clean_text(
            sample_row["employment_status"]
        )
        if sample_row is not None
        else employment_options[0]
    )

    employment_status = st.selectbox(
        "Employment Status",
        employment_options,
        index=safe_index(
            employment_options,
            employment_value
        ),
        key=f"employment_status_{input_widget_key}"
    )

    purpose_options = get_sorted_options(
        train_data,
        "loan_purpose"
    )

    purpose_value = (
        clean_text(
            sample_row["loan_purpose"]
        )
        if sample_row is not None
        else purpose_options[0]
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        purpose_options,
        index=safe_index(
            purpose_options,
            purpose_value
        ),
        key=f"loan_purpose_{input_widget_key}"
    )

    grade_options = get_sorted_options(
        train_data,
        "grade_subgrade"
    )

    grade_value = (
        clean_text(
            sample_row["grade_subgrade"]
        )
        if sample_row is not None
        else grade_options[0]
    )

    grade_subgrade = st.selectbox(
        "Grade Subgrade",
        grade_options,
        index=safe_index(
            grade_options,
            grade_value
        ),
        key=f"grade_subgrade_{input_widget_key}"
    )

input_data = pd.DataFrame({
    "annual_income": [
        annual_income
    ],
    "debt_to_income_ratio": [
        debt_to_income_ratio
    ],
    "credit_score": [
        credit_score
    ],
    "loan_amount": [
        loan_amount
    ],
    "interest_rate": [
        interest_rate
    ],
    "gender": [
        gender
    ],
    "marital_status": [
        marital_status
    ],
    "education_level": [
        education_level
    ],
    "employment_status": [
        employment_status
    ],
    "loan_purpose": [
        loan_purpose
    ],
    "grade_subgrade": [
        grade_subgrade
    ]
})

display_input_dataframe(
    dataframe=input_data,
    title="View Raw Input Data",
    expanded=False
)

actual_label = None

if (
    sample_row is not None
    and "loan_paid_back" in sample_row.index
    and pd.notna(
        sample_row["loan_paid_back"]
    )
):
    actual_label = int(
        sample_row["loan_paid_back"]
    )

    display_actual_label(
        actual_label
    )

try:
    input_encoded = preprocess_input(
        input_data=input_data,
        resources=resources
    )

except Exception as error:

    display_section_error(
        "Input preprocessing",
        error
    )

    st.stop()

prediction_tab, feature_analysis_tab, explanation_tab = (
    st.tabs(
        [
            "Prediction",
            "Feature Analysis",
            "Model Explanation"
        ]
    )
)

with prediction_tab:
    st.header(
        "Selected Model Prediction"
    )

    st.write(
        f"Current selected model: **{model_display_name}**"
    )

    if st.button(
        "Predict Loan Payback",
        type="primary",
        key="main_prediction_button",
        disabled=selected_model_object is None
    ):
        try:
            (
                default_probability,
                payback_probability
            ) = predict_model(
                prediction_type=prediction_type,
                selected_model=selected_model,
                input_encoded=input_encoded,
                resources=resources
            )

            validate_probabilities(
                default_probability,
                payback_probability
            )

            display_prediction_result(
                title=f"{model_display_name} Prediction",
                default_probability=default_probability,
                payback_probability=payback_probability,
                threshold=threshold,
                model_name=model_display_name,
                actual_label=actual_label
            )

        except Exception as error:
            display_section_error(
                f"{model_display_name} prediction",
                error
            )

    if prediction_type == "Original":
        display_input_dataframe(
            dataframe=input_encoded,
            title="View Original Encoded and Scaled Input",
            expanded=False
        )

    elif prediction_type == "RFE":
        try:
            input_rfe = prepare_rfe_input(
                input_encoded,
                resources
            )

            display_input_dataframe(
                dataframe=input_rfe,
                title="View RFE-Selected Input",
                expanded=False
            )

        except Exception as error:
            display_section_error(
                "RFE input preparation",
                error
            )

    else:
        try:
            component_table = (
                create_pca_component_table(
                    input_encoded,
                    resources
                )
            )

            with st.expander(
                "View PCA Component Values",
                expanded=False
            ):

                st.dataframe(
                    component_table,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:
            display_section_error(
                "PCA input transformation",
                error
            )

with feature_analysis_tab:
    st.header(
        "Feature Selection and Dimensionality Reduction"
    )

    st.write(
        "This section presents the experimental findings "
        "from RFE feature selection and PCA dimensionality "
        "reduction. Prediction is performed only in the "
        "Prediction tab using the sidebar configuration."
    )

    rfecv_tab, pca_tab = st.tabs(
        [
            "RFE",
            "PCA"
        ]
    )

    with rfecv_tab:
        st.subheader(
            "Recursive Feature Elimination"
        )

        st.write(
            "RFE recursively removes less important "
            "features and uses cross-validation to identify "
            "the feature subset with the highest mean "
            "ROC-AUC."
        )

        try:
            rfe_summary = get_rfe_summary(
                resources
            )

            rfe_metric_col1, rfe_metric_col2, (
                rfe_metric_col3
            ) = st.columns(3)

            with rfe_metric_col1:
                st.metric(
                    "Original Features",
                    rfe_summary[
                        "original_feature_count"
                    ]
                )

            with rfe_metric_col2:
                st.metric(
                    "Selected Features",
                    rfe_summary[
                        "selected_feature_count"
                    ]
                )

            with rfe_metric_col3:
                st.metric(
                    "Features Removed",
                    rfe_summary[
                        "removed_feature_count"
                    ]
                )

            st.info(
                get_rfe_interpretation(
                    resources
                )
            )

            with st.expander(
                "View RFE Summary",
                expanded=False
            ):
                rfe_summary_table = (
                    create_rfe_summary_table(
                        resources
                    )
                )

                st.dataframe(
                    rfe_summary_table,
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader(
                "Mean ROC-AUC by Number of Features"
            )

            st.write(
                "The chart shows how predictive performance "
                "changes as RFE retains different numbers "
                "of features."
            )

            rfe_auc_chart_data = (
                create_rfe_auc_chart_data()
            )

            if rfe_auc_chart_data.empty:
                st.warning(
                    "RFE feature-selection results are "
                    "not available."
                )

            else:
                st.line_chart(
                    rfe_auc_chart_data,
                    use_container_width=True
                )
                
            rfe_auc_display_table = (
                rfe_auc_chart_data
                .reset_index()
            )

            st.dataframe(
                rfe_auc_display_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Number of Features": st.column_config.NumberColumn(
                        "Number of Features",
                        format="%d"
                    ),
                    "Mean ROC-AUC": st.column_config.NumberColumn(
                        "Mean ROC-AUC",
                        format="%.6f"
                    )
                }
            )

            st.subheader(
                "Selected Features"
            )

            rfe_feature_table = (
                create_rfe_feature_table(
                    resources
                )
            )

            with st.expander(
                "View Selected Feature List",
                expanded=False
            ):
                st.dataframe(
                    rfe_feature_table,
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader(
                "Models Trained with RFE Features"
            )

            st.write(
                "This table compares the evaluation results "
                "of the four models trained using the "
                "RFE-selected feature set."
            )

            rfe_model_comparison = (
                create_rfe_model_comparison_table()
            )

            if rfe_model_comparison.empty:
                st.warning(
                    "RFE model-comparison results are "
                    "not available."
                )

            else:
                st.dataframe(
                    rfe_model_comparison,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:
            display_section_error(
                "RFE analysis",
                error
            )

    with pca_tab:
        st.subheader(
            "Principal Component Analysis"
        )

        st.write(
            "PCA transforms correlated original features "
            "into a smaller set of uncorrelated principal "
            "components while retaining as much data "
            "variance as possible."
        )

        try:
            pca_summary = get_pca_summary(
                resources
            )

            pca_metric_col1, pca_metric_col2, (
                pca_metric_col3
            ) = st.columns(3)

            with pca_metric_col1:
                st.metric(
                    "Original Features",
                    pca_summary[
                        "original_dimension"
                    ]
                )


            with pca_metric_col2:
                st.metric(
                    "PCA Components",
                    pca_summary[
                        "pca_dimension"
                    ]
                )


            with pca_metric_col3:
                if (
                    pca_summary[
                        "total_explained_variance"
                    ]
                    is None
                ):
                    total_variance_display = (
                        "Not available"
                    )

                else:
                    total_variance_display = (
                        f"{pca_summary['total_explained_variance']:.2%}"
                    )

                st.metric(
                    "Variance Retained",
                    total_variance_display
                )


            st.info(
                get_pca_interpretation(
                    resources
                )
            )

            with st.expander(
                "View PCA Summary",
                expanded=False
            ):

                pca_summary_table = (
                    create_pca_summary_table(
                        resources
                    )
                )

                st.dataframe(
                    pca_summary_table,
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader(
                "Explained Variance"
            )

            st.write(
                "Individual explained variance shows how "
                "much information each component captures. "
                "Cumulative variance shows the total "
                "information retained after adding each "
                "component."
            )

            variance_chart_data = (
                create_pca_variance_chart_data(
                    resources
                )
            )

            if variance_chart_data.empty:
                st.warning(
                    "PCA explained-variance results are "
                    "not available."
                )

            else:
                st.line_chart(
                    variance_chart_data,
                    use_container_width=True
                )


            formatted_variance_table = (
                create_formatted_variance_table(
                    resources
                )
            )

            with st.expander(
                "View Explained Variance Table",
                expanded=False
            ):
                st.dataframe(
                    formatted_variance_table,
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader(
                "Principal Component Loadings"
            )

            st.write(
                "Loadings indicate which original features "
                "contribute most strongly to each principal "
                "component. Positive and negative signs "
                "represent opposite directions within the "
                "component, not whether a feature is good "
                "or bad."
            )

            loading_summary = (
                create_pca_loading_summary(
                    resources,
                    top_n=3
                )
            )

            st.dataframe(
                loading_summary,
                use_container_width=True,
                hide_index=True
            )

            with st.expander(
                "View Current Borrower's PCA Component Values",
                expanded=False
            ):

                component_table = (
                    create_pca_component_table(
                        input_encoded,
                        resources
                    )
                )

                st.dataframe(
                    component_table,
                    use_container_width=True,
                    hide_index=True
                )

            st.subheader(
                "Models Trained with PCA Components"
            )

            st.write(
                "This table compares the four models after "
                "the original 23 features were transformed "
                "into 9 principal components."
            )

            pca_model_comparison = (
                create_pca_model_comparison_by_model()
            )

            if pca_model_comparison.empty:
                st.warning(
                    "PCA model-comparison results are "
                    "not available."
                )

            else:
                st.dataframe(
                    pca_model_comparison,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:
            display_section_error(
                "PCA analysis",
                error
            )
        

with explanation_tab:

    st.header(
        "Local Prediction Explanation"
    )

    st.write(
        "SHAP and LIME explain how the borrower features "
        "influence the current model prediction."
    )


    if (
        prediction_type != "Original"
        or selected_model not in [
            "Logistic Regression",
            "GAM",
            "EBM",
            "BLR"
        ]
    ):

        st.warning(
            "SHAP and LIME are currently available only "
            "for the original Logistic Regression, GAM, EBM "
            "and BLR models. Select Prediction Type = Original."
        )


    else:

        shap_tab, lime_tab = st.tabs(
            [
                "SHAP",
                "LIME"
            ]
        )

        with shap_tab:

            st.subheader(
                "SHAP Local Explanation"
            )


            shap_setting_col1, shap_setting_col2, shap_setting_col3 = (
                st.columns(3)
            )


            with shap_setting_col1:

                default_shap_background = (
                    50
                    if selected_model in ["EBM", "BLR"]
                    else 500
                )

                shap_background_size = st.slider(
                    "SHAP Background Sample Size",
                    min_value=50,
                    max_value=1000,
                    value=default_shap_background,
                    step=50
                )


            with shap_setting_col2:

                shap_feature_count = st.slider(
                    "Number of SHAP Features",
                    min_value=5,
                    max_value=min(
                        20,
                        len(model_columns)
                    ),
                    value=min(
                        10,
                        len(model_columns)
                    ),
                    step=1
                )

            with shap_setting_col3:

                if selected_model in ["EBM", "BLR"]:

                    shap_kernel_samples = st.slider(
                        "Kernel SHAP Samples",
                        min_value=100,
                        max_value=1000,
                        value=200,
                        step=100
                    )

                else:

                    shap_kernel_samples = 200

                    st.caption(
                        "Kernel SHAP samples are used only "
                        "for EBM and BLR."
                    )


            if st.button(
                "Generate SHAP Explanation",
                type="primary",
                key="shap_button"
            ):

                try:

                    with st.spinner(
                        "Generating SHAP explanation..."
                    ):

                        shap_explanation = (
                            calculate_shap_explanation(
                                selected_model=selected_model,
                                input_encoded=input_encoded,
                                resources=resources,
                                background_size=(
                                    shap_background_size
                                ),
                                kernel_samples=(
                                    shap_kernel_samples
                                )
                            )
                        )


                    shap_table = create_shap_table(
                        shap_explanation=shap_explanation,
                        resources=resources,
                        top_n=shap_feature_count
                    )


                    st.subheader(
                        "SHAP Feature Contributions"
                    )


                    st.dataframe(
                        shap_table[
                            [
                                "Feature",
                                "SHAP Contribution",
                                "Effect"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )


                    st.write(
                        "**SHAP Waterfall Plot**"
                    )

                    waterfall_figure = create_shap_waterfall_plot(
                        shap_explanation,
                        max_display=shap_feature_count
                    )

                    st.pyplot(
                        waterfall_figure,
                        width="stretch"
                    )

                    plt.close(
                        waterfall_figure
                    )

                    st.info(
                        "Positive SHAP contributions support "
                        "loan payback. Negative contributions "
                        "support loan default."
                    )


                except Exception as error:

                    display_section_error(
                        "SHAP explanation",
                        error
                    )

        with lime_tab:

            st.subheader(
                "LIME Local Explanation"
            )


            lime_setting_col1, lime_setting_col2 = (
                st.columns(2)
            )


            with lime_setting_col1:

                lime_background_size = st.slider(
                    "LIME Background Sample Size",
                    min_value=500,
                    max_value=5000,
                    value=3000,
                    step=500
                )


                lime_feature_count = st.slider(
                    "Number of LIME Features",
                    min_value=5,
                    max_value=min(
                        20,
                        len(model_columns)
                    ),
                    value=min(
                        10,
                        len(model_columns)
                    ),
                    step=1
                )


            with lime_setting_col2:

                lime_sample_count = st.slider(
                    "LIME Perturbation Samples",
                    min_value=1000,
                    max_value=10000,
                    value=5000,
                    step=1000
                )


            if st.button(
                "Generate LIME Explanation",
                type="primary",
                key="lime_button"
            ):

                try:

                    with st.spinner(
                        "Generating LIME explanation..."
                    ):

                        lime_explanation = (
                            calculate_lime_explanation(
                                selected_model=selected_model,
                                input_encoded=input_encoded,
                                resources=resources,
                                background_size=(
                                    lime_background_size
                                ),
                                number_of_features=(
                                    lime_feature_count
                                ),
                                number_of_samples=(
                                    lime_sample_count
                                )
                            )
                        )


                    lime_table = create_lime_table(
                        lime_explanation=lime_explanation,
                        label=1
                    )


                    st.subheader(
                        "LIME Feature Contributions"
                    )


                    st.dataframe(
                        lime_table[
                            [
                                "Feature Condition",
                                "LIME Contribution",
                                "Effect"
                            ]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )


                    lime_figure = create_lime_bar_plot(
                        lime_explanation=lime_explanation,
                        label=1
                    )


                    st.pyplot(
                        lime_figure,
                        use_container_width=True
                    )


                    plt.close(
                        lime_figure
                    )


                    st.info(
                        "Positive LIME contributions support "
                        "loan payback. Negative contributions "
                        "support loan default."
                    )


                except Exception as error:

                    display_section_error(
                        "LIME explanation",
                        error
                    )


st.markdown("---")

st.caption(
    "Final Year Project — Loan Payback Prediction "
    "Using Interpretable Models"
)
