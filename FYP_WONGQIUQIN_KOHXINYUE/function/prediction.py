import numpy as np
import pandas as pd

from function.rfe import prepare_rfe_input
from function.pca import (
    transform_pca_input,
    get_pca_feature_names
)


SUPPORTED_MODELS = [
    "Logistic Regression",
    "GAM",
    "EBM",
    "BLR"
]


SUPPORTED_PREDICTION_TYPES = [
    "Original",
    "RFE",
    "PCA"
]

def predict_standard_model(
    model,
    model_input
):
    probabilities = model.predict_proba(
        model_input
    )

    probabilities = np.asarray(
        probabilities
    )

    if probabilities.ndim == 1:

        payback_probability = float(
            probabilities.reshape(-1)[0]
        )

        default_probability = (
            1 - payback_probability
        )

    elif (
        probabilities.ndim == 2
        and probabilities.shape[1] >= 2
    ):

        default_probability = float(
            probabilities[0, 0]
        )

        payback_probability = float(
            probabilities[0, 1]
        )

    else:

        raise ValueError(
            "The model returned an unexpected "
            "probability output."
        )

    return (
        default_probability,
        payback_probability
    )

def predict_gam_model(
    model,
    model_input
):
    payback_probabilities = model.predict_proba(
        model_input
    )

    payback_probabilities = np.asarray(
        payback_probabilities
    )

    if (
        payback_probabilities.ndim == 2
        and payback_probabilities.shape[1] >= 2
    ):

        default_probability = float(
            payback_probabilities[0, 0]
        )

        payback_probability = float(
            payback_probabilities[0, 1]
        )

    else:

        payback_probability = float(
            payback_probabilities.reshape(-1)[0]
        )

        default_probability = (
            1 - payback_probability
        )

    return (
        default_probability,
        payback_probability
    )

def predict_blr_model(
    model,
    model_input,
    feature_names=None
):
    if not isinstance(
        model,
        dict
    ):
        raise TypeError(
            "The BLR model must be stored as a dictionary."
        )

    if "alpha_mean" not in model:
        raise KeyError(
            "BLR model does not contain 'alpha_mean'."
        )

    if "beta_mean" not in model:
        raise KeyError(
            "BLR model does not contain 'beta_mean'."
        )

    alpha_mean = float(
        np.asarray(
            model["alpha_mean"]
        ).reshape(-1)[0]
    )

    beta_mean = np.asarray(
        model["beta_mean"]
    ).reshape(-1)

    if isinstance(
        model_input,
        pd.DataFrame
    ):

        if feature_names is not None:

            missing_features = [
                feature
                for feature in feature_names
                if feature not in model_input.columns
            ]

            if missing_features:
                raise KeyError(
                    "The following BLR input features "
                    f"are missing: {missing_features}"
                )

            input_array = model_input[
                feature_names
            ].values.astype(float)

        else:

            input_array = (
                model_input
                .values
                .astype(float)
            )

    else:

        input_array = np.asarray(
            model_input,
            dtype=float
        )

    if input_array.ndim == 1:
        input_array = input_array.reshape(
            1,
            -1
        )

    if input_array.shape[1] != len(
        beta_mean
    ):
        raise ValueError(
            "BLR input feature count does not match "
            "the number of saved beta coefficients. "
            f"Input features: {input_array.shape[1]}, "
            f"beta coefficients: {len(beta_mean)}."
        )

    logit = (
        alpha_mean
        + np.dot(
            input_array,
            beta_mean
        )
    )

    logit = np.clip(
        logit,
        -500,
        500
    )

    payback_probability = float(
        (
            1 / (
                1 + np.exp(-logit)
            )
        ).reshape(-1)[0]
    )

    default_probability = (
        1 - payback_probability
    )

    return (
        default_probability,
        payback_probability
    )

def get_model_collection(
    prediction_type,
    resources
):
    if prediction_type == "Original":
        return resources.get(
            "models",
            {}
        )

    if prediction_type == "RFE":
        return resources.get(
            "rfe_models",
            {}
        )

    if prediction_type == "PCA":
        return resources.get(
            "pca_models",
            {}
        )

    raise ValueError(
        f"Unsupported prediction type: {prediction_type}"
    )

def prepare_model_input(
    prediction_type,
    input_encoded,
    resources
):
    if prediction_type == "Original":

        model_columns = list(
            resources["model_columns"]
        )

        return (
            input_encoded[
                model_columns
            ].copy(),
            model_columns
        )

    if prediction_type == "RFE":

        input_rfe = prepare_rfe_input(
            input_encoded,
            resources
        )

        rfe_feature_names = list(
            input_rfe.columns
        )

        return (
            input_rfe,
            rfe_feature_names
        )

    if prediction_type == "PCA":

        _, input_pca_df = transform_pca_input(
                input_encoded,
                resources
            )

        pca_feature_names = (
            get_pca_feature_names(
                resources
            )
        )

        input_pca_df = input_pca_df[
            pca_feature_names
        ]

        return (
            input_pca_df,
            pca_feature_names
        )

    raise ValueError(
        f"Unsupported prediction type: {prediction_type}"
    )

def predict_model(
    prediction_type,
    selected_model,
    input_encoded,
    resources
):
    if prediction_type not in (
        SUPPORTED_PREDICTION_TYPES
    ):
        raise ValueError(
            f"Unsupported prediction type: {prediction_type}"
        )

    if selected_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {selected_model}"
        )

    model_collection = get_model_collection(
        prediction_type,
        resources
    )

    model = model_collection.get(
        selected_model
    )

    if model is None:

        display_name = get_model_display_name(
            prediction_type,
            selected_model
        )

        raise ValueError(
            f"{display_name} model could not be loaded. "
            "Check whether the corresponding .pkl file exists."
        )

    (
        model_input,
        feature_names
    ) = prepare_model_input(
        prediction_type,
        input_encoded,
        resources
    )

    if selected_model == "GAM":

        return predict_gam_model(
            model,
            model_input
        )

    if selected_model == "BLR":

        return predict_blr_model(
            model,
            model_input,
            feature_names
        )

    return predict_standard_model(
        model,
        model_input
    )

def get_model_display_name(
    prediction_type,
    selected_model
):
    """
    Create a readable model name.
    """

    if prediction_type == "Original":
        return selected_model

    return (
        f"{selected_model} + {prediction_type}"
    )

def get_prediction_label(
    default_probability,
    threshold=0.5
):
    """
    Convert probability into a readable prediction.
    """

    if float(
        default_probability
    ) >= threshold:

        return "Loan Default"

    return "Loan Paid Back"

def create_prediction_result(
    prediction_type,
    selected_model,
    default_probability,
    payback_probability,
    threshold=0.5,
    actual_label=None
):
    """
    Create a prediction summary table.
    """

    display_name = get_model_display_name(
        prediction_type,
        selected_model
    )

    prediction_label = get_prediction_label(
        default_probability,
        threshold
    )

    result_data = {
        "Model": [
            display_name
        ],
        "Prediction": [
            prediction_label
        ],
        "Default Probability": [
            float(default_probability)
        ],
        "Payback Probability": [
            float(payback_probability)
        ]
    }

    if actual_label is not None:

        result_data[
            "Actual Label"
        ] = [
            int(actual_label)
        ]

    return pd.DataFrame(
        result_data
    )