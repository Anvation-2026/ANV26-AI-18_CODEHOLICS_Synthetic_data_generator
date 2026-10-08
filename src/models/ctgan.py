from sdv.single_table import CTGANSynthesizer


def create_ctgan(metadata):
    """
    Create a CTGAN synthesizer.
    """

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=300,
        verbose=True
    )

    return synthesizer


def train_ctgan(
    synthesizer,
    train_data
):
    """
    Train the CTGAN synthesizer.
    """

    synthesizer.fit(train_data)

    return synthesizer


def generate_ctgan_data(
    synthesizer,
    num_rows
):
    """
    Generate synthetic rows using CTGAN.
    """

    synthetic_data = synthesizer.sample(
        num_rows=num_rows
    )

    return synthetic_data