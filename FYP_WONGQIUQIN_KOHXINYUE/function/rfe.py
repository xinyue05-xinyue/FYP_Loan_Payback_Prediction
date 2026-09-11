import os
from networkx import display
import pandas as pd


RFE_SUMMARY_PATH = (
    "results/rfe_summary.csv"
)

RFE_SELECTION_RESULTS_PATH = (
    "results/rfe_feature_selection_results.csv"
)

RFE_RANKING_PATH = (
    "results/rfe_feature_ranking.csv"
)

RFE_MODEL_COMPARISON_PATH = (
    "results/rfe_models_comparison.csv"
)


def load_rfe_csv(
    file_path
):
    if not os.path.exists(file_path):
        return pd.DataFrame()

    try:
        return pd.read_csv(
            file_path
        )

    except Exception as error:
        raise ValueError(
            f"Unable to load RFE result file: "
            f"{file_path}. Error: {error}"
        ) from error

def prepare_rfe_input(
    input_encoded,
    resources
):
    selected_features = resources.get(
        "rfe_selected_features"
    )

    if selected_features is None:
        raise ValueError(
            "RFE-selected features could not be loaded."
        )

    selected_features = list(
        selected_features
    )

    missing_features = [
        feature
        for feature in selected_features
        if feature not in input_encoded.columns
    ]

    if missing_features:
        raise KeyError(
            "The following RFE-selected features are "
            f"missing from the processed input: "
            f"{missing_features}"
        )

    return input_encoded[
        selected_features
    ].copy()

def get_rfe_summary(
    resources
):
    model_columns = list(
        resources["model_columns"]
    )

    selected_features = resources.get(
        "rfe_selected_features"
    )

    if selected_features is None:
        raise ValueError(
            "RFE-selected features could not be loaded."
        )

    selected_features = list(
        selected_features
    )

    retained_all_features = (
        len(selected_features)
        == len(model_columns)
        and set(selected_features)
        == set(model_columns)
    )

    summary_results = load_rfe_csv(
        RFE_SUMMARY_PATH
    )

    training_time = None

    if not summary_results.empty:

        summary_mapping = dict(
            zip(
                summary_results["Item"],
                summary_results["Value"]
            )
        )

        training_time = summary_mapping.get(
            "RFE Training Time"
        )

    return {
        "original_feature_count":
            len(model_columns),

        "selected_feature_count":
            len(selected_features),

        "removed_feature_count":
            len(model_columns)
            - len(selected_features),

        "selected_features":
            selected_features,

        "retained_all_features":
            retained_all_features,

        "training_time":
            training_time
    }


def create_rfe_summary_table(
    resources
):
    summary = get_rfe_summary(
        resources
    )

    training_time = summary[
        "training_time"
    ]

    if training_time is None:
        formatted_training_time = "Not available"

    else:
        formatted_training_time = (
            f"{training_time:.2f} seconds"
        )

    return pd.DataFrame({
        "Measure": [
            "Original Number of Features",
            "Selected Number of Features",
            "Removed Number of Features",
            "RFE Search Time"
        ],

        "Value": [
            summary["original_feature_count"],
            summary["selected_feature_count"],
            summary["removed_feature_count"],
            formatted_training_time
        ]
    })


def create_rfe_feature_table(
    resources
):
    summary = get_rfe_summary(
        resources
    )

    selected_features = summary[
        "selected_features"
    ]

    return pd.DataFrame({
        "No.": range(
            1,
            len(selected_features) + 1
        ),

        "Selected Feature":
            selected_features
    })


def get_rfe_selection_results():
    results_df = load_rfe_csv(
        RFE_SELECTION_RESULTS_PATH
    )

    if results_df.empty:
        return results_df

    expected_columns = [
        "Number of Features",
        "Mean ROC-AUC",
        "Std ROC-AUC"
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in results_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The RFE selection-results file is missing "
            f"these columns: {missing_columns}"
        )

    results_df = results_df[
        expected_columns
    ].copy()

    results_df = results_df.sort_values(
        by="Number of Features"
    ).reset_index(
        drop=True
    )

    return results_df



def create_rfe_auc_chart_data():
    results_df = get_rfe_selection_results()

    if results_df.empty:
        return results_df

    chart_df = results_df[
        [
            "Number of Features",
            "Mean ROC-AUC"
        ]
    ].copy()

    chart_df = chart_df.set_index(
        "Number of Features"
    )

    return chart_df


def get_rfe_best_result():
    results_df = get_rfe_selection_results()

    if results_df.empty:
        return None

    best_index = results_df[
        "Mean ROC-AUC"
    ].idxmax()

    best_row = results_df.loc[
        best_index
    ]

    return {
        "number_of_features":
            int(
                best_row[
                    "Number of Features"
                ]
            ),

        "mean_roc_auc":
            float(
                best_row[
                    "Mean ROC-AUC"
                ]
            ),

        "std_roc_auc":
            float(
                best_row[
                    "Std ROC-AUC"
                ]
            )
    }



def get_rfe_model_comparison():
    comparison_df = load_rfe_csv(
        RFE_MODEL_COMPARISON_PATH
    )

    if comparison_df.empty:
        return comparison_df

    numeric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Log Loss",
        "Brier Score",
        "Training Time"
    ]

    for column in numeric_columns:

        if column in comparison_df.columns:

            comparison_df[column] = (
                pd.to_numeric(
                    comparison_df[column],
                    errors="coerce"
                )
            )

    return comparison_df


def create_rfe_model_comparison_table():
    comparison_df = get_rfe_model_comparison()

    if comparison_df.empty:
        return comparison_df

    formatted_df = comparison_df.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Log Loss",
        "Brier Score"
    ]

    for column in metric_columns:

        if column in formatted_df.columns:

            formatted_df[column] = (
                formatted_df[column]
                .round(4)
            )

    if "Training Time" in formatted_df.columns:

        formatted_df[
            "Training Time"
        ] = (
            formatted_df[
                "Training Time"
            ]
            .round(2)
        )

        formatted_df = formatted_df.rename(
            columns={
                "Training Time":
                    "Training Time (Seconds)"
            }
        )

    return formatted_df


def get_rfe_interpretation(
    resources
):
    summary = get_rfe_summary(
        resources
    )

    best_result = get_rfe_best_result()

    if best_result is None:

        return (
            "RFE results are unavailable because the "
            "feature-selection result file could not be loaded."
        )

    best_feature_count = best_result[
        "number_of_features"
    ]

    best_auc = best_result[
        "mean_roc_auc"
    ]

    if summary["retained_all_features"]:

        return (
            f"RFE selected all "
            f"{summary['selected_feature_count']} features. "
            f"The complete feature set achieved the highest "
            f"mean cross-validation ROC-AUC of "
            f"{best_auc:.4f}. Therefore, RFE did not "
            f"reduce the dimensionality of the dataset."
        )

    return (
        f"RFE reduced the feature set from "
        f"{summary['original_feature_count']} to "
        f"{best_feature_count} features. The selected subset "
        f"achieved a mean cross-validation ROC-AUC of "
        f"{best_auc:.4f}."
    )


def get_available_rfe_models(
    resources
):
    rfe_models = resources.get(
        "rfe_models",
        {}
    )

    return [
        model_name
        for model_name, model
        in rfe_models.items()
        if model is not None
    ]