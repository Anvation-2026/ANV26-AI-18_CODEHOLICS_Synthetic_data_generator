from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


def train_churn_model(
    data,
    feature_columns,
    categorical_features,
    numerical_features
):
    """
    Train a Random Forest churn classifier.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        data[feature_columns],
        data["churn"]
    )

    return model


def evaluate_churn_model(
    model,
    test_data,
    feature_columns
):
    """
    Evaluate a churn model using accuracy and F1.
    """

    predictions = model.predict(
        test_data[feature_columns]
    )

    accuracy = accuracy_score(
        test_data["churn"],
        predictions
    )

    f1 = f1_score(
        test_data["churn"],
        predictions
    )

    return {
        "accuracy": float(accuracy),
        "f1": float(f1)
    }