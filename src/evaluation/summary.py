def create_evaluation_summary(
    model_name,
    evaluation_results
):
    """
    Convert detailed evaluation results into
    a compact summary dictionary.
    """

    correlation_similarity = evaluation_results[
        "correlation"
    ]["correlation_similarity"]

    exact_match_rate = evaluation_results[
        "exact_match"
    ]["exact_match_rate"]

    privacy_distance_ratio = evaluation_results[
        "privacy_distance"
    ]["distance_ratio"]

    return {
        "model": model_name,
        "correlation_similarity": round(
            correlation_similarity, 4
        ),
        "exact_match_rate": round(
            exact_match_rate, 4
        ),
        "privacy_distance_ratio": round(
            privacy_distance_ratio, 4
        ),
    }
import pandas as pd


def load_final_results(file_path):
    """
    Load the final model evaluation summary.
    """

    return pd.read_csv(file_path)