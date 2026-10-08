from sdv.single_table import GaussianCopulaSynthesizer


def create_gaussian_copula(metadata):
    """
    Create a Gaussian Copula synthesizer.
    """

    synthesizer = GaussianCopulaSynthesizer(
        metadata
    )

    return synthesizer


def train_gaussian_copula(
    synthesizer,
    train_data
):
    """
    Train the Gaussian Copula synthesizer.
    """

    synthesizer.fit(train_data)

    return synthesizer


def generate_synthetic_data(
    synthesizer,
    num_rows
):
    """
    Generate synthetic rows.
    """

    synthetic_data = synthesizer.sample(
        num_rows=num_rows
    )

    return synthetic_data