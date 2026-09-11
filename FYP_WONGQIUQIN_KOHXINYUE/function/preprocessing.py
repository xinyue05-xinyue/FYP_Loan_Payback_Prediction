import numpy as np
import pandas as pd


EDUCATION_MAPPING = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Bachelor's": 2,
    "Master": 3,
    "Master's": 3,
    "PhD": 4,
    "Ph.D": 4,
    "Doctorate": 4
}


GRADE_MAPPING = {
    "A1": 1,
    "A2": 2,
    "A3": 3,
    "A4": 4,
    "A5": 5,

    "B1": 6,
    "B2": 7,
    "B3": 8,
    "B4": 9,
    "B5": 10,

    "C1": 11,
    "C2": 12,
    "C3": 13,
    "C4": 14,
    "C5": 15,

    "D1": 16,
    "D2": 17,
    "D3": 18,
    "D4": 19,
    "D5": 20,

    "E1": 21,
    "E2": 22,
    "E3": 23,
    "E4": 24,
    "E5": 25,

    "F1": 26,
    "F2": 27,
    "F3": 28,
    "F4": 29,
    "F5": 30
}


NUMERIC_COLUMNS = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount",
    "interest_rate",
    "education_level",
    "grade_subgrade"
]


NOMINAL_COLUMNS = [
    "gender",
    "marital_status",
    "employment_status",
    "loan_purpose"
]


TEXT_COLUMNS = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade"
]

def clean_text_columns(dataframe):
    """
    Remove leading and trailing spaces from categorical columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Raw borrower data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with cleaned text columns.
    """

    cleaned_data = dataframe.copy()

    for column in TEXT_COLUMNS:

        if column in cleaned_data.columns:

            cleaned_data[column] = (
                cleaned_data[column]
                .astype(str)
                .str.strip()
            )

    return cleaned_data


def encode_ordinal_columns(dataframe):
    """
    Convert education level and grade subgrade into
    ordered numerical values.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Borrower data with text-based ordinal columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with encoded ordinal columns.
    """

    encoded_data = dataframe.copy()

    encoded_data["education_level"] = (
        encoded_data["education_level"]
        .map(EDUCATION_MAPPING)
    )

    encoded_data["grade_subgrade"] = (
        encoded_data["grade_subgrade"]
        .map(GRADE_MAPPING)
    )

    encoded_data["education_level"] = (
        encoded_data["education_level"]
        .fillna(0)
    )

    encoded_data["grade_subgrade"] = (
        encoded_data["grade_subgrade"]
        .fillna(1)
    )

    return encoded_data


def handle_numeric_missing_values(
    dataframe,
    train_data
):
    """
    Convert numerical columns to numeric format and fill
    missing values using medians from the training dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Borrower data being preprocessed.

    train_data : pandas.DataFrame
        Original training dataset used to obtain median values.

    Returns
    -------
    pandas.DataFrame
        DataFrame with numerical missing values handled.
    """

    processed_data = dataframe.copy()

    raw_numeric_columns = [
        "annual_income",
        "debt_to_income_ratio",
        "credit_score",
        "loan_amount",
        "interest_rate"
    ]

    for column in raw_numeric_columns:

        processed_data[column] = pd.to_numeric(
            processed_data[column],
            errors="coerce"
        )

        training_median = pd.to_numeric(
            train_data[column],
            errors="coerce"
        ).median()

        processed_data[column] = (
            processed_data[column]
            .fillna(training_median)
        )

    return processed_data


def one_hot_encode_columns(
    dataframe,
    model_columns
):
    """
    One-hot encode nominal categorical variables and ensure
    that the output columns match the model training columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Borrower data after ordinal encoding.

    model_columns : list
        Final feature column order used during model training.

    Returns
    -------
    pandas.DataFrame
        One-hot encoded DataFrame with matching columns.
    """

    encoded_data = pd.get_dummies(
        dataframe,
        columns=NOMINAL_COLUMNS
    )

    encoded_data = encoded_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    encoded_data = encoded_data.astype(float)

    return encoded_data


def scale_numeric_columns(
    dataframe,
    scaler
):
    """
    Apply the saved StandardScaler to numerical columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Encoded borrower data.

    scaler : sklearn scaler object
        Saved scaler used during model training.

    Returns
    -------
    pandas.DataFrame
        Encoded and scaled borrower data.
    """

    scaled_data = dataframe.copy()

    scaled_data[NUMERIC_COLUMNS] = scaler.transform(
        scaled_data[NUMERIC_COLUMNS]
    )

    return scaled_data


def validate_processed_data(dataframe):
    """
    Check that the processed input contains no NaN or
    infinite values.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Fully preprocessed borrower data.

    Raises
    ------
    ValueError
        If NaN or infinite values are detected.
    """

    if dataframe.isnull().sum().sum() > 0:

        missing_columns = (
            dataframe.isnull()
            .sum()
        )

        missing_columns = missing_columns[
            missing_columns > 0
        ]

        raise ValueError(
            "NaN values remain after preprocessing. "
            f"Affected columns: {missing_columns.to_dict()}"
        )

    if not np.isfinite(
        dataframe.values
    ).all():

        raise ValueError(
            "Infinite values were detected after preprocessing."
        )


def preprocess_input(
    input_data,
    resources
):
    """
    Apply all preprocessing steps required by the trained models.

    Processing steps:
    1. Remove unnecessary columns.
    2. Clean categorical text.
    3. Encode ordinal variables.
    4. Handle numerical missing values.
    5. One-hot encode nominal variables.
    6. Reindex columns to match model training.
    7. Scale numerical features.
    8. Validate final data.

    Parameters
    ----------
    input_data : pandas.DataFrame
        Raw borrower input from the Streamlit application.

    resources : dict
        Dictionary returned by load_resources().
        It must contain:
        - resources["train_data"]
        - resources["model_columns"]
        - resources["scaler"]

    Returns
    -------
    pandas.DataFrame
        Fully encoded and scaled input ready for prediction.
    """

    processed_data = input_data.copy()

    train_data = resources["train_data"]
    model_columns = list(
        resources["model_columns"]
    )
    scaler = resources["scaler"]

    columns_to_remove = [
        "id",
        "loan_paid_back"
    ]

    removable_columns = [
        column
        for column in columns_to_remove
        if column in processed_data.columns
    ]

    if removable_columns:

        processed_data = processed_data.drop(
            columns=removable_columns
        )

    processed_data = clean_text_columns(
        processed_data
    )

    processed_data = encode_ordinal_columns(
        processed_data
    )


    processed_data = handle_numeric_missing_values(
        processed_data,
        train_data
    )


    processed_data = one_hot_encode_columns(
        processed_data,
        model_columns
    )

    processed_data = scale_numeric_columns(
        processed_data,
        scaler
    )

    validate_processed_data(
        processed_data
    )

    return processed_data