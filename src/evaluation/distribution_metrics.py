from scipy.stats import ks_2samp


def calculate_ks_metrics(real_data, synthetic_data, numerical_columns):
    """
    Compare real and synthetic numerical distributions
    using the Kolmogorov-Smirnov statistic.
    """

    results = {}

    for column in numerical_columns:
        statistic, p_value = ks_2samp(
            real_data[column],
            synthetic_data[column]
        )

        results[column] = {
            "ks_statistic": float(statistic),
            "p_value": float(p_value)
        }

    return results