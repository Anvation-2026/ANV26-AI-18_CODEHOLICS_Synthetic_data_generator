from src.evaluation.distribution_metrics import calculate_ks_metrics
from src.evaluation.correlation_metrics import calculate_correlation_similarity
from src.evaluation.privacy_metrics import (
    calculate_nearest_neighbor_privacy,
    check_exact_match_privacy,
)


def evaluate_synthetic_data(
    train_data,
    synthetic_data,
    numerical_columns
):
    """
    Run the core utility and privacy checks
    for a synthetic dataset.
    """

    ks_results = calculate_ks_metrics(
        train_data,
        synthetic_data,
        numerical_columns
    )

    correlation_results = calculate_correlation_similarity(
        train_data,
        synthetic_data,
        numerical_columns
    )

    exact_match_results = check_exact_match_privacy(
        train_data,
        synthetic_data
    )

    privacy_results = calculate_nearest_neighbor_privacy(
        train_data,
        synthetic_data,
        numerical_columns
    )

    return {
        "ks_metrics": ks_results,
        "correlation": correlation_results,
        "exact_match": exact_match_results,
        "privacy_distance": privacy_results,
    }