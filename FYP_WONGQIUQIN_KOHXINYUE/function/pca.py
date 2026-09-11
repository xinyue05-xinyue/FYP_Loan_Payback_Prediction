import os

import numpy as np
import pandas as pd


PCA_SUMMARY_PATH = (
    "results/pca_summary.csv"
)

PCA_EXPLAINED_VARIANCE_PATH = (
    "results/pca_explained_variance.csv"
)

PCA_MODEL_COMPARISON_PATH = (
    "results/pca_models_comparison.csv"
)


def load_pca_csv(
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
            f"Unable to load PCA result file: "
            f"{file_path}. Error: {error}"
        ) from error

def get_pca_feature_names(
    resources
):
    pca_transformer = resources.get(
        "pca_transformer"
    )

    if pca_transformer is None:
        raise ValueError(
            "PCA transformer could not be loaded."
        )

    if hasattr(
        pca_transformer,
        "n_components_"
    ):
        number_of_components = int(
            pca_transformer.n_components_
        )

    elif hasattr(
        pca_transformer,
        "components_"
    ):
        number_of_components = int(
            pca_transformer.components_.shape[0]
        )

    else:
        raise ValueError(
            "The number of PCA components could not "
            "be determined."
        )

    return [
        f"PC{component_number}"
        for component_number in range(
            1,
            number_of_components + 1
        )
    ]


def transform_pca_input(
    input_encoded,
    resources
):
    pca_transformer = resources.get(
        "pca_transformer"
    )

    if pca_transformer is None:
        raise ValueError(
            "PCA transformer could not be loaded."
        )

    model_columns = list(
        resources["model_columns"]
    )

    missing_columns = [
        column
        for column in model_columns
        if column not in input_encoded.columns
    ]

    if missing_columns:
        raise KeyError(
            "The following model columns are missing "
            f"from the processed input: "
            f"{missing_columns}"
        )

    input_for_pca = input_encoded[
        model_columns
    ].copy()

    input_pca = pca_transformer.transform(
        input_for_pca
    )

    input_pca = np.asarray(
        input_pca
    )

    pca_columns = get_pca_feature_names(
        resources
    )

    if input_pca.shape[1] != len(
        pca_columns
    ):
        pca_columns = [
            f"PC{component_number}"
            for component_number in range(
                1,
                input_pca.shape[1] + 1
            )
        ]

    input_pca_df = pd.DataFrame(
        input_pca,
        columns=pca_columns,
        index=input_encoded.index
    )

    return (
        input_pca,
        input_pca_df
    )


def get_pca_summary(
    resources
):
    pca_transformer = resources.get(
        "pca_transformer"
    )

    if pca_transformer is None:
        raise ValueError(
            "PCA transformer could not be loaded."
        )

    model_columns = list(
        resources["model_columns"]
    )

    pca_feature_names = get_pca_feature_names(
        resources
    )

    total_explained_variance = None

    if hasattr(
        pca_transformer,
        "explained_variance_ratio_"
    ):
        total_explained_variance = float(
            np.asarray(
                pca_transformer
                .explained_variance_ratio_
            ).sum()
        )

    summary_results = load_pca_csv(
        PCA_SUMMARY_PATH
    )

    transformation_time = None

    if not summary_results.empty:

        required_columns = {
            "Item",
            "Value"
        }

        if required_columns.issubset(
            summary_results.columns
        ):
            summary_mapping = dict(
                zip(
                    summary_results["Item"],
                    summary_results["Value"]
                )
            )

            transformation_time = (
                summary_mapping.get(
                    "PCA Transformation Time"
                )
            )

            if transformation_time is not None:
                transformation_time = float(
                    transformation_time
                )

    dimension_reduction = (
        len(model_columns)
        - len(pca_feature_names)
    )

    reduction_percentage = (
        dimension_reduction
        / len(model_columns)
        * 100
    )

    return {
        "original_dimension":
            len(model_columns),

        "pca_dimension":
            len(pca_feature_names),

        "dimension_reduction":
            dimension_reduction,

        "reduction_percentage":
            reduction_percentage,

        "total_explained_variance":
            total_explained_variance,

        "transformation_time":
            transformation_time
    }


def create_pca_summary_table(
    resources
):
    summary = get_pca_summary(
        resources
    )

    explained_variance = summary[
        "total_explained_variance"
    ]

    if explained_variance is None:
        explained_variance_display = (
            "Not available"
        )

    else:
        explained_variance_display = (
            f"{explained_variance * 100:.2f}%"
        )

    transformation_time = summary[
        "transformation_time"
    ]

    if transformation_time is None:
        transformation_time_display = (
            "Not available"
        )

    else:
        transformation_time_display = (
            f"{transformation_time:.4f} seconds"
        )

    return pd.DataFrame({
        "Measure": [
            "Original Number of Features",
            "Number of PCA Components",
            "Dimensions Removed",
            "Dimension Reduction",
            "Total Explained Variance",
            "PCA Transformation Time"
        ],

        "Value": [
            summary["original_dimension"],
            summary["pca_dimension"],
            summary["dimension_reduction"],
            (
                f"{summary['reduction_percentage']:.2f}%"
            ),
            explained_variance_display,
            transformation_time_display
        ]
    })


def create_pca_component_table(
    input_encoded,
    resources
):
    _, input_pca_df = transform_pca_input(
        input_encoded,
        resources
    )

    if len(input_pca_df) == 1:

        component_table = (
            input_pca_df
            .T
            .reset_index()
        )

        component_table.columns = [
            "Principal Component",
            "Component Value"
        ]

        component_table[
            "Component Value"
        ] = (
            component_table[
                "Component Value"
            ]
            .round(4)
        )

        return component_table

    component_table = (
        input_pca_df
        .reset_index(
            drop=True
        )
    )

    component_table.index = (
        component_table.index + 1
    )

    component_table.index.name = (
        "Borrower"
    )

    return component_table.reset_index()


def create_explained_variance_table(
    resources
):
    saved_variance_df = load_pca_csv(
        PCA_EXPLAINED_VARIANCE_PATH
    )

    expected_columns = [
        "Principal Component",
        "Explained Variance Ratio",
        "Cumulative Explained Variance"
    ]

    if not saved_variance_df.empty:

        missing_columns = [
            column
            for column in expected_columns
            if column not in saved_variance_df.columns
        ]

        if not missing_columns:

            variance_df = saved_variance_df[
                expected_columns
            ].copy()

            return variance_df

    pca_transformer = resources.get(
        "pca_transformer"
    )

    if pca_transformer is None:
        raise ValueError(
            "PCA transformer could not be loaded."
        )

    if not hasattr(
        pca_transformer,
        "explained_variance_ratio_"
    ):
        raise ValueError(
            "The PCA transformer does not contain "
            "explained variance information."
        )

    variance_ratio = np.asarray(
        pca_transformer
        .explained_variance_ratio_
    )

    pca_columns = get_pca_feature_names(
        resources
    )

    return pd.DataFrame({
        "Principal Component":
            pca_columns,

        "Explained Variance Ratio":
            variance_ratio,

        "Cumulative Explained Variance":
            np.cumsum(
                variance_ratio
            )
    })


def create_formatted_variance_table(
    resources
):
    variance_df = (
        create_explained_variance_table(
            resources
        )
    )

    formatted_df = variance_df.copy()

    formatted_df[
        "Explained Variance (%)"
    ] = (
        formatted_df[
            "Explained Variance Ratio"
        ]
        * 100
    ).round(2)

    formatted_df[
        "Cumulative Variance (%)"
    ] = (
        formatted_df[
            "Cumulative Explained Variance"
        ]
        * 100
    ).round(2)

    return formatted_df[
        [
            "Principal Component",
            "Explained Variance (%)",
            "Cumulative Variance (%)"
        ]
    ]


def create_pca_variance_chart_data(
    resources
):
    variance_df = (
        create_explained_variance_table(
            resources
        )
    )

    chart_df = variance_df[
        [
            "Principal Component",
            "Explained Variance Ratio",
            "Cumulative Explained Variance"
        ]
    ].copy()

    chart_df[
        "Explained Variance (%)"
    ] = (
        chart_df[
            "Explained Variance Ratio"
        ]
        * 100
    )

    chart_df[
        "Cumulative Variance (%)"
    ] = (
        chart_df[
            "Cumulative Explained Variance"
        ]
        * 100
    )

    chart_df = chart_df[
        [
            "Principal Component",
            "Explained Variance (%)",
            "Cumulative Variance (%)"
        ]
    ]

    return chart_df.set_index(
        "Principal Component"
    )

def create_pca_loading_table(
    resources
):
    pca_transformer = resources.get(
        "pca_transformer"
    )

    if pca_transformer is None:
        raise ValueError(
            "PCA transformer could not be loaded."
        )

    if not hasattr(
        pca_transformer,
        "components_"
    ):
        raise ValueError(
            "The PCA transformer does not contain "
            "component loading information."
        )

    model_columns = list(
        resources["model_columns"]
    )

    component_names = get_pca_feature_names(
        resources
    )

    components = np.asarray(
        pca_transformer.components_
    )

    if components.shape[1] != len(
        model_columns
    ):
        raise ValueError(
            "The number of PCA loading columns does "
            "not match the model feature columns."
        )

    loading_df = pd.DataFrame(
        components,
        index=component_names,
        columns=model_columns
    )

    loading_df.index.name = (
        "Principal Component"
    )

    return loading_df.reset_index()

def create_pca_loading_summary(
    resources,
    top_n=3
):
    loading_df = (
        create_pca_loading_table(
            resources
        )
        .set_index(
            "Principal Component"
        )
    )

    summary_rows = []

    for component_name, row in (
        loading_df.iterrows()
    ):
        positive_features = (
            row.sort_values(
                ascending=False
            )
            .head(top_n)
        )

        negative_features = (
            row.sort_values(
                ascending=True
            )
            .head(top_n)
        )

        positive_text = ", ".join(
            [
                f"{feature} ({value:.3f})"
                for feature, value
                in positive_features.items()
            ]
        )

        negative_text = ", ".join(
            [
                f"{feature} ({value:.3f})"
                for feature, value
                in negative_features.items()
            ]
        )

        summary_rows.append({
            "Principal Component":
                component_name,

            "Strongest Positive Loadings":
                positive_text,

            "Strongest Negative Loadings":
                negative_text
        })

    return pd.DataFrame(
        summary_rows
    )


def get_pca_model_comparison():
    comparison_df = load_pca_csv(
        PCA_MODEL_COMPARISON_PATH
    )

    if comparison_df.empty:
        return comparison_df

    expected_columns = [
        "Metric",
        "LR + PCA",
        "BLR + PCA",
        "EBM + PCA",
        "GAM + PCA"
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in comparison_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The PCA model-comparison file is missing "
            f"these columns: {missing_columns}"
        )

    return comparison_df[
        expected_columns
    ].copy()

def create_pca_model_comparison_table():
    comparison_df = (
        get_pca_model_comparison()
    )

    if comparison_df.empty:
        return comparison_df

    formatted_df = comparison_df.copy()

    model_columns = [
        column
        for column in formatted_df.columns
        if column != "Metric"
    ]

    for column in model_columns:

        formatted_df[column] = pd.to_numeric(
            formatted_df[column],
            errors="coerce"
        )

    metric_rows = (
        formatted_df["Metric"]
        != "Number of Components"
    )

    formatted_df.loc[
        metric_rows,
        model_columns
    ] = (
        formatted_df.loc[
            metric_rows,
            model_columns
        ]
        .round(4)
    )

    component_rows = (
        formatted_df["Metric"]
        == "Number of Components"
    )

    formatted_df.loc[
        component_rows,
        model_columns
    ] = (
        formatted_df.loc[
            component_rows,
            model_columns
        ]
        .round(0)
    )

    return formatted_df


def create_pca_model_comparison_by_model():
    comparison_df = (
        get_pca_model_comparison()
    )

    if comparison_df.empty:
        return comparison_df

    transposed_df = (
        comparison_df
        .set_index(
            "Metric"
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "Model"
            }
        )
    )

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

        if column in transposed_df.columns:

            transposed_df[column] = (
                pd.to_numeric(
                    transposed_df[column],
                    errors="coerce"
                )
                .round(4)
            )

    if (
        "Training Time"
        in transposed_df.columns
    ):
        transposed_df[
            "Training Time"
        ] = (
            pd.to_numeric(
                transposed_df[
                    "Training Time"
                ],
                errors="coerce"
            )
            .round(2)
        )

        transposed_df = (
            transposed_df.rename(
                columns={
                    "Training Time":
                        "Training Time (Seconds)"
                }
            )
        )

    if (
        "Number of Components"
        in transposed_df.columns
    ):
        transposed_df[
            "Number of Components"
        ] = (
            pd.to_numeric(
                transposed_df[
                    "Number of Components"
                ],
                errors="coerce"
            )
            .astype(
                "Int64"
            )
        )

    return transposed_df

def get_pca_interpretation(
    resources
):
    summary = get_pca_summary(
        resources
    )

    explained_variance = summary[
        "total_explained_variance"
    ]

    if explained_variance is None:

        return (
            f"PCA reduced the input dimensions from "
            f"{summary['original_dimension']} features "
            f"to {summary['pca_dimension']} components. "
            f"Explained variance information is unavailable."
        )

    return (
        f"PCA reduced the input dimensions from "
        f"{summary['original_dimension']} features to "
        f"{summary['pca_dimension']} principal components, "
        f"representing a dimensionality reduction of "
        f"{summary['reduction_percentage']:.2f}%. "
        f"The retained components explain "
        f"{explained_variance * 100:.2f}% of the total "
        f"variance in the original feature space."
    )


def get_available_pca_models(
    resources
):
    pca_models = resources.get(
        "pca_models",
        {}
    )

    return [
        model_name
        for model_name, model
        in pca_models.items()
        if model is not None
    ]