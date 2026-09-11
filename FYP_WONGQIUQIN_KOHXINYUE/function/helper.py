import pandas as pd
import streamlit as st


def safe_index(
    options,
    value,
    default=0
):
    """
    Find the index of a value inside a list.

    This prevents the Streamlit selectbox from crashing
    when a value is not found in the available options.

    Parameters
    ----------
    options : list
        Available selectbox options.

    value : object
        Value whose index should be found.

    default : int, optional
        Index returned when the value is unavailable.

    Returns
    -------
    int
        Index of the value or the default index.
    """

    try:
        return options.index(value)

    except (
        ValueError,
        AttributeError
    ):
        return default


def clean_text(value):
    """
    Convert a value to clean text by removing leading
    and trailing spaces.

    Parameters
    ----------
    value : object
        Input value.

    Returns
    -------
    str
        Cleaned text value.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


def get_sorted_options(
    train_data,
    column_name
):
    """
    Get sorted unique values for a categorical feature.

    Parameters
    ----------
    train_data : pandas.DataFrame
        Original training dataset.

    column_name : str
        Name of the categorical column.

    Returns
    -------
    list
        Sorted unique category values.

    Raises
    ------
    KeyError
        If the requested column does not exist.
    """

    if column_name not in train_data.columns:
        raise KeyError(
            f"Column '{column_name}' was not found "
            "in the training dataset."
        )

    options = (
        train_data[column_name]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    options = sorted(
        options
    )

    return options


def format_actual_label(
    actual_label
):
    """
    Convert the numerical target value into a readable label.

    Parameters
    ----------
    actual_label : int or float
        Target value:
        - 0 = Loan Default
        - 1 = Loan Paid Back

    Returns
    -------
    str
        Readable target label.
    """

    if actual_label is None:
        return "Unknown"

    try:
        actual_label = int(
            actual_label
        )

    except (
        TypeError,
        ValueError
    ):
        return "Unknown"

    if actual_label == 1:
        return "Loan Paid Back"

    if actual_label == 0:
        return "Loan Default"

    return "Unknown"


def display_actual_label(
    actual_label
):
    """
    Display the actual label of a randomly selected borrower.

    Parameters
    ----------
    actual_label : int or float
        Actual target value from train.csv.
    """

    if actual_label is None:
        return

    readable_label = format_actual_label(
        actual_label
    )

    if int(actual_label) == 1:

        st.info(
            "Actual label from train.csv: "
            "1 — Loan Paid Back"
        )

    elif int(actual_label) == 0:

        st.info(
            "Actual label from train.csv: "
            "0 — Loan Default"
        )

    else:

        st.info(
            "Actual label from train.csv: "
            f"{actual_label} — {readable_label}"
        )


def display_prediction_result(
    title,
    default_probability,
    payback_probability,
    threshold=0.5,
    model_name=None,
    actual_label=None
):
    """
    Display prediction label, probabilities and summary table.

    Parameters
    ----------
    title : str
        Section title shown above the prediction.

    default_probability : float
        Predicted probability of loan default.

    payback_probability : float
        Predicted probability of loan payback.

    threshold : float, optional
        Classification threshold for loan default.

    model_name : str, optional
        Name of the prediction model.

    actual_label : int or float, optional
        Actual label from train.csv.

    Returns
    -------
    pandas.DataFrame
        Prediction summary table.
    """

    default_probability = float(
        default_probability
    )

    payback_probability = float(
        payback_probability
    )

    st.subheader(
        title
    )

    if default_probability >= threshold:

        prediction_label = (
            "Loan Default"
        )

        st.error(
            "Prediction: Loan Default"
        )

    else:

        prediction_label = (
            "Loan Paid Back"
        )

        st.success(
            "Prediction: Loan Paid Back"
        )

    probability_col1, probability_col2 = (
        st.columns(2)
    )

    with probability_col1:

        st.metric(
            "Default Probability",
            f"{default_probability:.2%}"
        )

    with probability_col2:

        st.metric(
            "Payback Probability",
            f"{payback_probability:.2%}"
        )

    result_data = {
        "Prediction": [
            prediction_label
        ],
        "Default Probability": [
            default_probability
        ],
        "Payback Probability": [
            payback_probability
        ]
    }

    if model_name is not None:

        result_data = {
            "Model": [
                model_name
            ],
            **result_data
        }

    if actual_label is not None:

        result_data[
            "Actual Label"
        ] = [
            int(actual_label)
        ]

        result_data[
            "Actual Outcome"
        ] = [
            format_actual_label(
                actual_label
            )
        ]

    result_df = pd.DataFrame(
        result_data
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    return result_df


def display_input_dataframe(
    dataframe,
    title,
    expanded=False
):
    """
    Display a DataFrame inside a Streamlit expander.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Data to display.

    title : str
        Expander title.

    expanded : bool, optional
        Whether the expander is opened initially.
    """

    with st.expander(
        title,
        expanded=expanded
    ):

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )


def validate_probabilities(
    default_probability,
    payback_probability,
    tolerance=1e-6
):
    """
    Validate model probability outputs.

    Parameters
    ----------
    default_probability : float
        Probability of default.

    payback_probability : float
        Probability of payback.

    tolerance : float, optional
        Accepted numerical difference from a total of 1.

    Raises
    ------
    ValueError
        If probabilities are outside the valid range or
        do not approximately sum to 1.
    """

    default_probability = float(
        default_probability
    )

    payback_probability = float(
        payback_probability
    )

    if not (
        0 <= default_probability <= 1
    ):
        raise ValueError(
            "Default probability must be "
            "between 0 and 1."
        )

    if not (
        0 <= payback_probability <= 1
    ):
        raise ValueError(
            "Payback probability must be "
            "between 0 and 1."
        )

    probability_total = (
        default_probability
        + payback_probability
    )

    if abs(
        probability_total - 1
    ) > tolerance:

        raise ValueError(
            "Default and payback probabilities "
            "do not sum to 1."
        )


def display_section_error(
    section_name,
    error
):
    """
    Display a consistent error message for a Streamlit section.

    Parameters
    ----------
    section_name : str
        Name of the section that failed.

    error : Exception
        Raised exception.
    """

    st.error(
        f"{section_name} failed."
    )

    st.exception(
        error
    )