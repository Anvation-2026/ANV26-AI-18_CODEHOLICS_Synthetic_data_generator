# Synthetic Data Generator with Privacy & Utility Validation

## Project ID

AI-04

## Problem

Generate realistic synthetic tabular data while retaining useful
analytical properties without reproducing sensitive source records.

## Objective

This project aims to generate synthetic tabular datasets and evaluate
whether the generated data preserves useful analytical properties while
reducing the risk of memorizing or reproducing records from the original
dataset.

## Synthetic Data Models

The project will compare multiple synthetic data generation approaches:

1. Gaussian Copula
2. CTGAN
3. TVAE

## Utility Evaluation

The synthetic datasets will be evaluated using:

- Distribution similarity
- Statistical similarity
- Correlation similarity
- Machine learning performance
- Train on Synthetic, Test on Real (TSTR)
- Train on Real, Test on Synthetic (TRTS)

## Privacy Evaluation

The project will evaluate:

- Exact record matching
- Duplicate records
- Nearest-neighbor similarity
- Training versus holdout similarity
- Memorization risk

## Final Demonstration

The final prototype will provide an interactive interface where a user can:

1. Upload a tabular dataset
2. Inspect the dataset
3. Select a synthetic data generation model
4. Generate synthetic records
5. Evaluate utility
6. Evaluate privacy risk
7. Compare different models
8. Download the synthetic dataset

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- SDV
- Matplotlib
- Seaborn
- Plotly
- Jupyter
- Streamlit

## Project Status

Development in progress.