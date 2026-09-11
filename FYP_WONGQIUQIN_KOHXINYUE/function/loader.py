import os
import joblib
import pandas as pd


def load_optional_model(file_path):
    if os.path.exists(file_path):
        return joblib.load(file_path)

    return None

def load_resources():
    lr_model = joblib.load(
        "models/lr_model.pkl"
    )

    gam_model = joblib.load(
        "models/gam_model.pkl"
    )

    ebm_model = joblib.load(
        "models/ebm_model.pkl"
    )

    blr_model = joblib.load(
        "models/blr_model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    model_columns = joblib.load(
        "models/model_columns.pkl"
    )

    rfe_selector = load_optional_model(
        "models/rfe_selector.pkl"
    )

    rfe_selected_features = load_optional_model(
        "models/rfe_selected_features.pkl"
    )

    pca_transformer = load_optional_model(
        "models/pca_transformer.pkl"
    )

    lr_rfe_model = load_optional_model(
        "models/lr_rfe_model.pkl"
    )

    gam_rfe_model = load_optional_model(
        "models/gam_rfe_model.pkl"
    )

    ebm_rfe_model = load_optional_model(
        "models/ebm_rfe_model.pkl"
    )

    blr_rfe_model = load_optional_model(
        "models/blr_rfe_model.pkl"
    )

    lr_pca_model = load_optional_model(
        "models/lr_pca_model.pkl"
    )

    gam_pca_model = load_optional_model(
        "models/gam_pca_model.pkl"
    )

    ebm_pca_model = load_optional_model(
        "models/ebm_pca_model.pkl"
    )

    blr_pca_model = load_optional_model(
        "models/blr_pca_model.pkl"
    )

    train_data = pd.read_csv(
        "data/train.csv"
    )

    if "id" in train_data.columns:

        train_data = train_data.drop(
            columns=["id"]
        )

    resources = {
        "models": {
            "Logistic Regression": lr_model,
            "GAM": gam_model,
            "EBM": ebm_model,
            "BLR": blr_model
        },

        "rfe_models": {
            "Logistic Regression": lr_rfe_model,
            "GAM": gam_rfe_model,
            "EBM": ebm_rfe_model,
            "BLR": blr_rfe_model
        },

        "pca_models": {
            "Logistic Regression": lr_pca_model,
            "GAM": gam_pca_model,
            "EBM": ebm_pca_model,
            "BLR": blr_pca_model
        },

        "scaler": scaler,

        "model_columns": model_columns,

        "train_data": train_data,

        "rfe_selector": rfe_selector,

        "rfe_selected_features":
            rfe_selected_features,

        "pca_transformer":
            pca_transformer
    }

    return resources