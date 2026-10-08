import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def calculate_exact_match_rate(
    train_data,
    synthetic_data
):
    """
    Calculate the percentage of synthetic rows
    that exactly match training rows.
    """

    train_records = set(
        map(tuple, train_data.to_numpy())
    )

    synthetic_records = set(
        map(tuple, synthetic_data.to_numpy())
    )

    exact_matches = train_records.intersection(
        synthetic_records
    )

    match_rate = (
        len(exact_matches)
        / len(synthetic_records)
    )

    return {
        "exact_matches": len(exact_matches),
        "exact_match_rate": float(match_rate)
    }


def calculate_nearest_neighbor_privacy(
    train_data,
    synthetic_data,
    numerical_columns
):
    """
    Compare synthetic-to-training nearest-neighbor
    distances with training-to-training distances.
    """

    scaler = StandardScaler()

    train_numeric = scaler.fit_transform(
        train_data[numerical_columns]
    )

    synthetic_numeric = scaler.transform(
        synthetic_data[numerical_columns]
    )

    synthetic_distances = pairwise_distances(
        synthetic_numeric,
        train_numeric,
        metric="euclidean"
    )

    synthetic_nearest = synthetic_distances.min(
        axis=1
    )

    training_distances = pairwise_distances(
        train_numeric,
        train_numeric,
        metric="euclidean"
    )

    np.fill_diagonal(
        training_distances,
        np.inf
    )

    training_nearest = training_distances.min(
        axis=1
    )

    distance_ratio = (
        synthetic_nearest.mean()
        / training_nearest.mean()
    )

    return {
        "synthetic_average_nearest_distance":
            float(synthetic_nearest.mean()),

        "synthetic_minimum_nearest_distance":
            float(synthetic_nearest.min()),

        "training_average_nearest_distance":
            float(training_nearest.mean()),

        "training_minimum_nearest_distance":
            float(training_nearest.min()),

        "distance_ratio":
            float(distance_ratio)
    }
def check_exact_match_privacy(train_data, synthetic_data):
    """
    Check whether synthetic records exactly match
    records from the training dataset.
    """

    result = calculate_exact_match_rate(
        train_data,
        synthetic_data
    )

    if result["exact_match_rate"] == 0:
        result["status"] = "No exact training-record matches found."
    else:
        result["status"] = "Exact training-record matches detected."

    return result