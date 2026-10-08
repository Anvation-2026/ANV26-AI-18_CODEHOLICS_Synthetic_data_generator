import numpy as np


def calculate_correlation_similarity(
    real_data,
    synthetic_data,
    numerical_columns
):
    """
    Calculate similarity between real and synthetic
    numerical correlation matrices.
    """

    real_corr = real_data[numerical_columns].corr()
    synthetic_corr = synthetic_data[numerical_columns].corr()

    difference = (
        real_corr - synthetic_corr
    ).abs()

    mask = ~np.eye(
        difference.shape[0],
        dtype=bool
    )

    mean_error = difference.values[mask].mean()

    similarity = 1 - mean_error

    return {
        "mean_absolute_correlation_error": float(mean_error),
        "correlation_similarity": float(similarity)
    }