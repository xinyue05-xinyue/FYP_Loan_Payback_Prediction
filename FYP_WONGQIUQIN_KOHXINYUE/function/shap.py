import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from function.preprocessing import preprocess_input


def prepare_shap_background(
    resources,
    sample_size=500,
    random_state=42
):
    """
    Prepare a sample of encoded and scaled training data
    to use as the SHAP background dataset.

    Parameters
    ----------
    resources : dict
        Dictionary returned by load_resources().

    sample_size : int, optional
        Number of training records used as SHAP background data.

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
            "SHAP background sample size must be greater than 0."
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


def ensure_valid_probability(
    probability
):
    """
    Convert model predictions into a clean one-dimensional
    probability array.

    Parameters
    ----------
    probability : array-like
        Predicted class-1 probabilities.

    Returns
    -------
    numpy.ndarray
        Valid probabilities between 0 and 1.
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

def create_gam_prediction_function(
    gam_model,
    model_columns
):
    """
    Create a prediction function that returns GAM payback
    probabilities for SHAP.

    Parameters
    ----------
    gam_model : trained LogisticGAM
        Saved GAM model.

    model_columns : list
        Feature order used during model training.

    Returns
    -------
    callable
        Function accepted by SHAP.
    """

    def gam_predict_payback(input_array):

        input_dataframe = pd.DataFrame(
            input_array,
            columns=model_columns
        )

        payback_probabilities = (
            gam_model.predict_proba(
                input_dataframe
            )
        )

        return np.asarray(
            payback_probabilities
        ).reshape(-1)

    return gam_predict_payback


def calculate_lr_shap(
    model,
    input_encoded,
    background_data
):
    """
    Calculate local SHAP values for Logistic Regression.

    Parameters
    ----------
    model : trained LogisticRegression
        Saved Logistic Regression model.

    input_encoded : pandas.DataFrame
        Fully encoded and scaled borrower input.

    background_data : pandas.DataFrame
        Encoded and scaled SHAP background sample.

    Returns
    -------
    shap.Explanation
        SHAP explanation for the current borrower.
    """

    explainer = shap.LinearExplainer(
        model,
        background_data
    )

    shap_explanation = explainer(
        input_encoded
    )

    return shap_explanation

def create_ebm_prediction_function(
    ebm_model,
    model_columns
):

    def ebm_predict_payback(input_array):

        input_df = pd.DataFrame(
            input_array,
            columns=model_columns
        )

        probability = ebm_model.predict_proba(
            input_df
        )[:, 1]

        return probability

    return ebm_predict_payback

def create_ebm_prediction_function(
    ebm_model,
    model_columns
):
    """
    Create a prediction function that returns EBM payback
    probabilities for SHAP.

    Parameters
    ----------
    ebm_model : trained ExplainableBoostingClassifier
        Saved EBM model.

    model_columns : list
        Feature order used during model training.

    Returns
    -------
    callable
        Prediction function accepted by SHAP.
    """

    def ebm_predict_payback(
        input_array
    ):

        input_dataframe = pd.DataFrame(
            input_array,
            columns=model_columns
        )

        probabilities = (
            ebm_model.predict_proba(
                input_dataframe
            )
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

        return ensure_valid_probability(
            payback_probabilities
        )

    return ebm_predict_payback


def create_blr_prediction_function(
    blr_model,
    model_columns
):
    """
    Create a prediction function that returns BLR payback
    probabilities using the saved posterior mean parameters.

    The saved BLR model is expected to be a dictionary
    containing:
    - alpha_mean
    - beta_mean

    Parameters
    ----------
    blr_model : dict
        Saved BLR model dictionary.

    model_columns : list
        Feature order used during model training.

    Returns
    -------
    callable
        Prediction function accepted by SHAP.
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

    def blr_predict_payback(
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

        return ensure_valid_probability(
            payback_probabilities
        )

    return blr_predict_payback

def calculate_gam_shap(
    model,
    input_encoded,
    background_data,
    model_columns
):
    """
    Calculate local SHAP values for GAM.

    A model-agnostic permutation explainer is used because
    LogisticGAM is not directly supported by every specialised
    SHAP explainer.

    Parameters
    ----------
    model : trained LogisticGAM
        Saved GAM model.

    input_encoded : pandas.DataFrame
        Fully encoded and scaled borrower input.

    background_data : pandas.DataFrame
        Encoded and scaled SHAP background sample.

    model_columns : list
        Feature order used during model training.

    Returns
    -------
    shap.Explanation
        SHAP explanation for the current borrower.
    """

    gam_prediction_function = (
        create_gam_prediction_function(
            model,
            model_columns
        )
    )

    explainer = shap.Explainer(
        gam_prediction_function,
        background_data,
        algorithm="permutation"
    )

    minimum_evaluations = (
        2 * len(model_columns)
        + 1
    )

    shap_explanation = explainer(
        input_encoded,
        max_evals=minimum_evaluations
    )

    return shap_explanation

def calculate_ebm_shap(
    model,
    input_encoded,
    background_data,
    model_columns,
    number_of_samples=200
):
    """
    Calculate a local SHAP explanation for EBM using
    KernelExplainer.

    Parameters
    ----------
    model : trained ExplainableBoostingClassifier
        Saved EBM model.

    input_encoded : pandas.DataFrame
        Fully encoded and scaled borrower input.

    background_data : pandas.DataFrame
        Encoded and scaled SHAP background sample.

    model_columns : list
        Feature order used during model training.

    number_of_samples : int, optional
        Number of Kernel SHAP samples.

    Returns
    -------
    shap.Explanation
        Local SHAP explanation for the borrower.
    """

    if number_of_samples <= 0:
        raise ValueError(
            "The number of SHAP samples must be "
            "greater than 0."
        )

    ebm_prediction_function = (
        create_ebm_prediction_function(
            ebm_model=model,
            model_columns=model_columns
        )
    )

    explainer = shap.KernelExplainer(
        ebm_prediction_function,
        background_data.to_numpy(
            dtype=float
        )
    )

    shap_values = explainer.shap_values(
        input_encoded.to_numpy(
            dtype=float
        ),
        nsamples=number_of_samples
    )

    shap_values = np.asarray(
        shap_values,
        dtype=float
    )

    if shap_values.ndim == 1:

        local_values = shap_values

    elif shap_values.ndim == 2:

        local_values = shap_values[0]

    elif shap_values.ndim == 3:
        local_values = shap_values[
            0,
            :,
            -1
        ]

    else:

        local_values = shap_values.reshape(
            -1
        )

    expected_value = float(
        np.asarray(
            explainer.expected_value
        ).reshape(-1)[0]
    )

    borrower_values = (
        input_encoded.iloc[0]
        .to_numpy(
            dtype=float
        )
        .reshape(-1)
    )

    local_explanation = shap.Explanation(
        values=local_values,
        base_values=expected_value,
        data=borrower_values,
        feature_names=model_columns
    )

    return shap.Explanation(
        values=np.asarray(
            local_explanation.values
        ).reshape(
            1,
            -1
        ),
        base_values=np.asarray([
            local_explanation.base_values
        ]),
        data=np.asarray(
            local_explanation.data
        ).reshape(
            1,
            -1
        ),
        feature_names=model_columns
    )

def calculate_blr_shap(
    model,
    input_encoded,
    background_data,
    model_columns,
    number_of_samples=200
):
    """
    Calculate a local SHAP explanation for BLR using
    KernelExplainer.

    Parameters
    ----------
    model : dict
        Saved BLR model containing alpha_mean and beta_mean.

    input_encoded : pandas.DataFrame
        Fully encoded and scaled borrower input.

    background_data : pandas.DataFrame
        Encoded and scaled SHAP background sample.

    model_columns : list
        Feature order used during model training.

    number_of_samples : int, optional
        Number of Kernel SHAP samples.

    Returns
    -------
    shap.Explanation
        Local SHAP explanation for the borrower.
    """

    if number_of_samples <= 0:
        raise ValueError(
            "The number of SHAP samples must be "
            "greater than 0."
        )

    blr_prediction_function = (
        create_blr_prediction_function(
            blr_model=model,
            model_columns=model_columns
        )
    )

    explainer = shap.KernelExplainer(
        blr_prediction_function,
        background_data.to_numpy(
            dtype=float
        )
    )

    shap_values = explainer.shap_values(
        input_encoded.to_numpy(
            dtype=float
        ),
        nsamples=number_of_samples
    )

    shap_values = np.asarray(
        shap_values,
        dtype=float
    )

    if shap_values.ndim == 1:

        local_values = shap_values

    elif shap_values.ndim == 2:

        local_values = shap_values[0]

    elif shap_values.ndim == 3:

        local_values = shap_values[
            0,
            :,
            -1
        ]

    else:

        local_values = shap_values.reshape(
            -1
        )

    expected_value = float(
        np.asarray(
            explainer.expected_value
        ).reshape(-1)[0]
    )

    borrower_values = (
        input_encoded.iloc[0]
        .to_numpy(
            dtype=float
        )
        .reshape(-1)
    )

    local_explanation = shap.Explanation(
        values=local_values,
        base_values=expected_value,
        data=borrower_values,
        feature_names=model_columns
    )

    return shap.Explanation(
        values=np.asarray(
            local_explanation.values
        ).reshape(
            1,
            -1
        ),
        base_values=np.asarray([
            local_explanation.base_values
        ]),
        data=np.asarray(
            local_explanation.data
        ).reshape(
            1,
            -1
        ),
        feature_names=model_columns
    )


def calculate_shap_explanation(
    selected_model,
    input_encoded,
    resources,
    background_size=500,
    random_state=42,
    kernel_samples=200
):
    """
    Generate a local SHAP explanation for the selected model.

    Supported models:
    - Logistic Regression
    - GAM
    - EBM
    - BLR

    Parameters
    ----------
    selected_model : str
        Model selected in the Streamlit sidebar.

    input_encoded : pandas.DataFrame
        Fully encoded and scaled borrower input.

    resources : dict
        Dictionary returned by load_resources().

    background_size : int, optional
        Number of training records initially sampled for
        SHAP background preparation.

    random_state : int, optional
        Random seed for reproducibility.

    kernel_samples : int, optional
        Number of samples used by KernelExplainer for
        EBM and BLR.

    Returns
    -------
    shap.Explanation
        Local SHAP explanation.
    """

    supported_models = [
        "Logistic Regression",
        "GAM",
        "EBM",
        "BLR"
    ]

    if selected_model not in supported_models:
        raise ValueError(
            "SHAP is available only for Logistic Regression, "
            "GAM, EBM and BLR."
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
            "The borrower input is missing model columns: "
            f"{missing_input_columns}"
        )

    input_for_shap = input_encoded[
        model_columns
    ].copy()

    background_data = (
        prepare_shap_background(
            resources=resources,
            sample_size=background_size,
            random_state=random_state
        )
    )

    missing_background_columns = [
        column
        for column in model_columns
        if column not in background_data.columns
    ]

    if missing_background_columns:
        raise ValueError(
            "The SHAP background data is missing columns: "
            f"{missing_background_columns}"
        )

    background_data = background_data[
        model_columns
    ].copy()

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        if (
            selected_model
            == "Logistic Regression"
        ):

            return calculate_lr_shap(
                model=model,
                input_encoded=input_for_shap,
                background_data=background_data
            )

        if selected_model == "GAM":

            return calculate_gam_shap(
                model=model,
                input_encoded=input_for_shap,
                background_data=background_data,
                model_columns=model_columns
            )

        if selected_model == "EBM":

            return calculate_ebm_shap(
                model=model,
                input_encoded=input_for_shap,
                background_data=background_data,
                model_columns=model_columns,
                number_of_samples=kernel_samples
            )

        return calculate_blr_shap(
            model=model,
            input_encoded=input_for_shap,
            background_data=background_data,
            model_columns=model_columns,
            number_of_samples=kernel_samples
        )

def extract_shap_values(
    shap_explanation,
    number_of_features
):
    """
    Extract one-dimensional SHAP values for one borrower.

    Parameters
    ----------
    shap_explanation : shap.Explanation
        SHAP explanation returned by an explainer.

    number_of_features : int
        Expected number of model features.

    Returns
    -------
    numpy.ndarray
        One SHAP contribution for each feature.
    """

    values = np.asarray(
        shap_explanation.values
    )

    if values.ndim == 1:

        shap_values = values

    elif values.ndim == 2:

        shap_values = values[0]

    elif values.ndim == 3:
        shap_values = values[
            0,
            :,
            -1
        ]

    else:

        shap_values = values.reshape(
            -1
        )

    if len(shap_values) < number_of_features:
        raise ValueError(
            "The SHAP explanation returned fewer values "
            "than the number of model features."
        )

    return shap_values[
        :number_of_features
    ]


def create_shap_table(
    shap_explanation,
    resources,
    top_n=10
):
    """
    Create a table showing the most important SHAP
    contributions for the current borrower.

    Parameters
    ----------
    shap_explanation : shap.Explanation
        Local SHAP explanation.

    resources : dict
        Dictionary returned by load_resources().

    top_n : int, optional
        Number of features to include.

    Returns
    -------
    pandas.DataFrame
        SHAP feature contribution table.
    """

    model_columns = list(
        resources["model_columns"]
    )

    shap_values = extract_shap_values(
        shap_explanation,
        len(model_columns)
    )

    shap_table = pd.DataFrame({
        "Feature": model_columns,
        "SHAP Contribution": shap_values,
        "Absolute Contribution": np.abs(
            shap_values
        )
    })

    shap_table["Effect"] = np.where(
        shap_table[
            "SHAP Contribution"
        ] >= 0,
        "Supports Payback",
        "Supports Default"
    )

    shap_table = shap_table.sort_values(
        "Absolute Contribution",
        ascending=False
    )

    top_n = min(
        top_n,
        len(shap_table)
    )

    return shap_table.head(
        top_n
    ).reset_index(
        drop=True
    )


def create_shap_waterfall_plot(
    shap_explanation,
    max_display=10
):
    """
    Create a SHAP waterfall figure for the current borrower.

    Parameters
    ----------
    shap_explanation : shap.Explanation
        Local SHAP explanation.

    max_display : int, optional
        Maximum number of features displayed.

    Returns
    -------
    matplotlib.figure.Figure
        Waterfall plot figure.
    """

    plt.close(
        "all"
    )

    shap.plots.waterfall(
        shap_explanation[0],
        max_display=max_display,
        show=False
    )

    figure = plt.gcf()

    figure.set_size_inches(
        11,
        7
    )

    figure.tight_layout()

    return figure

