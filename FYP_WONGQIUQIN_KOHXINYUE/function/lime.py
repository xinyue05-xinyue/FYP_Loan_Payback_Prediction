import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

from function.preprocessing import preprocess_input

def ensure_valid_probability(
    probability
):
    """
    Convert model predictions into a valid one-dimensional
    probability array.

    Parameters
    ----------
    probability : array-like
        Predicted class-1 probabilities.

    Returns
    -------
    numpy.ndarray
        Probabilities between 0 and 1.
    """

    probability = np.asarray(
        probability,
        dtype=float
    ).reshape(-1)

    probability = np.nan_to_num(
        probability,
        nan=0.5,
        posinf=1.0,
        neginf=0.0
    )

    probability = np.clip(
        probability,
        1e-6,
        1 - 1e-6
    )

    return probability


def prepare_lime_background(
    resources,
    sample_size=3000,
    random_state=42
):
    """
    Prepare encoded and scaled training data for LIME.

    Parameters
    ----------
    resources : dict
        Dictionary returned by load_resources().

    sample_size : int, optional
        Number of training rows used by LIME.

    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Encoded and scaled background dataset.
    """

    train_data = resources[
        "train_data"
    ]

    if train_data is None:
        raise ValueError(
            "Training data could not be loaded."
        )

    if sample_size <= 0:
        raise ValueError(
            "LIME background sample size must be "
            "greater than 0."
        )

    actual_sample_size = min(
        sample_size,
        len(train_data)
    )

    background_raw = train_data.sample(
        n=actual_sample_size,
        random_state=random_state
    ).copy()

    background_encoded = preprocess_input(
        background_raw,
        resources
    )

    return background_encoded


def create_lr_lime_predict_function(
    lr_model,
    model_columns
):
    """
    Create a Logistic Regression probability function
    compatible with LIME.

    LIME requires two probability columns:

    column 0 = Loan Default
    column 1 = Loan Paid Back
    """

    def lr_predict_proba(
        input_array
    ):

        input_dataframe = pd.DataFrame(
            input_array,
            columns=model_columns
        )

        probabilities = lr_model.predict_proba(
            input_dataframe
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if (
            probabilities.ndim != 2
            or probabilities.shape[1] < 2
        ):
            raise ValueError(
                "Logistic Regression predict_proba() "
                "must return two probability columns."
            )

        payback_probabilities = (
            probabilities[:, 1]
        )

        payback_probabilities = (
            ensure_valid_probability(
                payback_probabilities
            )
        )

        return np.column_stack(
            [
                1.0 - payback_probabilities,
                payback_probabilities
            ]
        )

    return lr_predict_proba


def create_gam_lime_predict_function(
    gam_model,
    model_columns
):
    """
    Create a GAM probability function compatible with LIME.

    LIME requires two probability columns:

    column 0 = Loan Default
    column 1 = Loan Paid Back
    """

    def gam_predict_proba(
        input_array
    ):

        input_dataframe = pd.DataFrame(
            input_array,
            columns=model_columns
        )

        payback_probabilities = (
            gam_model.predict_proba(
                input_dataframe
            )
        )

        payback_probabilities = (
            ensure_valid_probability(
                payback_probabilities
            )
        )

        return np.column_stack(
            [
                1.0 - payback_probabilities,
                payback_probabilities
            ]
        )

    return gam_predict_proba

def create_ebm_lime_predict_function(
    ebm_model,
    model_columns
):
    """
    Create an EBM probability function compatible with LIME.

    LIME requires two probability columns:

    column 0 = Loan Default
    column 1 = Loan Paid Back
    """

    def ebm_predict_proba(
        input_array
    ):

        input_dataframe = pd.DataFrame(
            input_array,
            columns=model_columns
        )

        probabilities = ebm_model.predict_proba(
            input_dataframe
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if (
            probabilities.ndim != 2
            or probabilities.shape[1] < 2
        ):
            raise ValueError(
                "EBM predict_proba() must return "
                "two probability columns."
            )

        payback_probabilities = (
            probabilities[:, 1]
        )

        payback_probabilities = (
            ensure_valid_probability(
                payback_probabilities
            )
        )

        return np.column_stack(
            [
                1.0 - payback_probabilities,
                payback_probabilities
            ]
        )

    return ebm_predict_proba


def create_blr_lime_predict_function(
    blr_model,
    model_columns
):
    """
    Create a BLR probability function compatible with LIME.

    The saved BLR model must be a dictionary containing:

    - alpha_mean
    - beta_mean

    LIME requires two probability columns:

    column 0 = Loan Default
    column 1 = Loan Paid Back
    """

    if not isinstance(
        blr_model,
        dict
    ):
        raise TypeError(
            "The BLR model must be a dictionary "
            "containing alpha_mean and beta_mean."
        )

    required_keys = {
        "alpha_mean",
        "beta_mean"
    }

    missing_keys = (
        required_keys.difference(
            blr_model.keys()
        )
    )

    if missing_keys:
        raise KeyError(
            "BLR model is missing the following keys: "
            f"{sorted(missing_keys)}"
        )

    alpha_mean = float(
        np.asarray(
            blr_model["alpha_mean"]
        ).squeeze()
    )

    beta_mean = np.asarray(
        blr_model["beta_mean"],
        dtype=float
    ).reshape(-1)

    if len(beta_mean) != len(model_columns):
        raise ValueError(
            "The number of BLR coefficients does not match "
            "the number of model columns. "
            f"BLR has {len(beta_mean)} coefficients, but "
            f"{len(model_columns)} columns were provided."
        )

    def blr_predict_proba(
        input_array
    ):

        input_dataframe = pd.DataFrame(
            input_array,
            columns=model_columns
        )

        input_values = input_dataframe.to_numpy(
            dtype=float
        )

        if (
            input_values.shape[1]
            != len(beta_mean)
        ):
            raise ValueError(
                f"BLR expects {len(beta_mean)} features, "
                f"but received "
                f"{input_values.shape[1]} features."
            )

        linear_predictor = (
            alpha_mean
            + input_values @ beta_mean
        )

        linear_predictor = np.clip(
            linear_predictor,
            -500,
            500
        )

        payback_probabilities = (
            1.0
            / (
                1.0
                + np.exp(
                    -linear_predictor
                )
            )
        )

        payback_probabilities = (
            ensure_valid_probability(
                payback_probabilities
            )
        )

        return np.column_stack(
            [
                1.0 - payback_probabilities,
                payback_probabilities
            ]
        )

    return blr_predict_proba


def create_lime_explainer(
    background_data,
    model_columns,
    random_state=42
):
    """
    Create a LIME tabular explainer.

    Parameters
    ----------
    background_data : pandas.DataFrame
        Encoded and scaled training background data.

    model_columns : list
        Feature names used during model training.

    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    LimeTabularExplainer
        Configured LIME explainer.
    """

    lime_explainer = LimeTabularExplainer(
        training_data=background_data.to_numpy(
            dtype=float
        ),
        feature_names=model_columns,
        class_names=[
            "Loan Default",
            "Loan Paid Back"
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=random_state
    )

    return lime_explainer

def get_lime_prediction_function(
    selected_model,
    model,
    model_columns
):
    """
    Return the correct LIME prediction function for the
    selected model.
    """

    if selected_model == "Logistic Regression":

        return create_lr_lime_predict_function(
            lr_model=model,
            model_columns=model_columns
        )

    if selected_model == "GAM":

        return create_gam_lime_predict_function(
            gam_model=model,
            model_columns=model_columns
        )

    if selected_model == "EBM":

        return create_ebm_lime_predict_function(
            ebm_model=model,
            model_columns=model_columns
        )

    if selected_model == "BLR":

        return create_blr_lime_predict_function(
            blr_model=model,
            model_columns=model_columns
        )

    raise ValueError(
        f"LIME does not support the selected model: "
        f"{selected_model}"
    )


def calculate_lime_explanation(
    selected_model,
    input_encoded,
    resources,
    background_size=3000,
    number_of_features=10,
    number_of_samples=5000,
    random_state=42
):
    """
    Generate a local LIME explanation for the selected model.

    Supported models:
    - Logistic Regression
    - GAM
    - EBM
    - BLR

    Parameters
    ----------
    selected_model : str
        Selected model name.

    input_encoded : pandas.DataFrame
        Fully encoded and scaled borrower input.

    resources : dict
        Dictionary returned by load_resources().

    background_size : int, optional
        Number of training rows used to build the explainer.

    number_of_features : int, optional
        Number of features included in the explanation.

    number_of_samples : int, optional
        Number of perturbed samples generated by LIME.

    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    lime.explanation.Explanation
        LIME explanation for the current borrower.
    """

    supported_models = [
        "Logistic Regression",
        "GAM",
        "EBM",
        "BLR"
    ]

    if selected_model not in supported_models:

        raise ValueError(
            "LIME is currently available only for "
            "Logistic Regression, GAM, EBM and BLR."
        )

    if background_size <= 0:

        raise ValueError(
            "LIME background size must be "
            "greater than 0."
        )

    if number_of_features <= 0:

        raise ValueError(
            "The number of LIME features must be "
            "greater than 0."
        )

    if number_of_samples <= 0:

        raise ValueError(
            "The number of LIME samples must be "
            "greater than 0."
        )

    model = resources[
        "models"
    ].get(
        selected_model
    )

    if model is None:

        raise ValueError(
            f"{selected_model} model could not be loaded."
        )

    model_columns = list(
        resources["model_columns"]
    )

    missing_input_columns = [
        column
        for column in model_columns
        if column not in input_encoded.columns
    ]

    if missing_input_columns:

        raise ValueError(
            "The encoded borrower input is missing "
            "the following model columns: "
            f"{missing_input_columns}"
        )

    input_for_lime = input_encoded[
        model_columns
    ].copy()

    background_data = prepare_lime_background(
        resources=resources,
        sample_size=background_size,
        random_state=random_state
    )

    missing_background_columns = [
        column
        for column in model_columns
        if column not in background_data.columns
    ]

    if missing_background_columns:

        raise ValueError(
            "The LIME background data is missing "
            "the following model columns: "
            f"{missing_background_columns}"
        )

    background_data = background_data[
        model_columns
    ].copy()

    lime_explainer = create_lime_explainer(
        background_data=background_data,
        model_columns=model_columns,
        random_state=random_state
    )

    prediction_function = (
        get_lime_prediction_function(
            selected_model=selected_model,
            model=model,
            model_columns=model_columns
        )
    )

    number_of_features = min(
        number_of_features,
        len(model_columns)
    )

    explanation = lime_explainer.explain_instance(
        data_row=(
            input_for_lime.iloc[0]
            .to_numpy(
                dtype=float
            )
        ),
        predict_fn=prediction_function,
        num_features=number_of_features,
        num_samples=number_of_samples,
        labels=(1,)
    )

    return explanation

def create_lime_table(
    lime_explanation,
    label=1
):
    """
    Create a DataFrame containing LIME contributions.

    Parameters
    ----------
    lime_explanation : lime.explanation.Explanation
        LIME explanation returned by explain_instance().

    label : int, optional
        Class to explain.
        1 represents Loan Paid Back.

    Returns
    -------
    pandas.DataFrame
        LIME feature contribution table.
    """

    lime_results = lime_explanation.as_list(
        label=label
    )

    lime_table = pd.DataFrame(
        lime_results,
        columns=[
            "Feature Condition",
            "LIME Contribution"
        ]
    )

    lime_table[
        "Absolute Contribution"
    ] = np.abs(
        lime_table[
            "LIME Contribution"
        ]
    )

    lime_table["Effect"] = np.where(
        lime_table[
            "LIME Contribution"
        ] >= 0,
        "Supports Payback",
        "Supports Default"
    )

    lime_table = lime_table.sort_values(
        "Absolute Contribution",
        ascending=False
    ).reset_index(
        drop=True
    )

    return lime_table

def create_lime_bar_plot(
    lime_explanation,
    label=1,
    figure_width=10,
    figure_height=6
):
    """
    Create a horizontal bar plot from a LIME explanation.

    Parameters
    ----------
    lime_explanation : lime.explanation.Explanation
        LIME explanation.

    label : int, optional
        Explained class.

    figure_width : int or float, optional
        Plot width.

    figure_height : int or float, optional
        Plot height.

    Returns
    -------
    matplotlib.figure.Figure
        LIME contribution plot.
    """

    lime_table = create_lime_table(
        lime_explanation=lime_explanation,
        label=label
    )

    plot_table = lime_table.sort_values(
        "LIME Contribution",
        ascending=True
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            figure_height
        )
    )

    axis.barh(
        plot_table[
            "Feature Condition"
        ],
        plot_table[
            "LIME Contribution"
        ]
    )

    axis.axvline(
        x=0,
        linewidth=1
    )

    axis.set_xlabel(
        "LIME Contribution"
    )

    axis.set_ylabel(
        "Feature Condition"
    )

    axis.set_title(
        "LIME Local Feature Contributions"
    )

    figure.tight_layout()

    return figure

def extract_lime_feature_names(
    lime_table,
    model_columns,
    top_n=5
):
    """
    Match LIME feature conditions to original feature names.

    This function is useful when comparing LIME results
    with SHAP feature rankings.
    """

    top_conditions = (
        lime_table[
            "Feature Condition"
        ]
        .head(top_n)
        .astype(str)
        .tolist()
    )

    matched_features = []

    sorted_model_columns = sorted(
        model_columns,
        key=len,
        reverse=True
    )

    for condition in top_conditions:

        matched_feature = None

        for feature in sorted_model_columns:

            if feature in condition:

                matched_feature = feature
                break

        if (
            matched_feature is not None
            and matched_feature not in matched_features
        ):

            matched_features.append(
                matched_feature
            )

    return matched_features