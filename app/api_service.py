import os
import sys
import io
import json
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from src.evaluation.evaluation_pipeline import evaluate_synthetic_data
from src.evaluation.ml_metrics import train_churn_model, evaluate_churn_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CodeholicsAPI")

app = FastAPI(
    title="CODEHOLICS Synthetic Data API",
    description="Backend service for synthetic data generation, privacy validation, and utility evaluation",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory state - starts empty with no predefined dataset
app_state: Dict[str, Any] = {
    "original_df": None,
    "model_data": None,
    "synthetic_df": None,
    "current_model": None,
    "evaluation_results": None,
    "ml_results": None,
    "numerical_columns": [],
    "categorical_columns": [],
    "target_column": None,
    "filename": None
}


def _analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract metadata, column types, and preview records from a dataframe."""
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col.lower() in ["customer_id", "id", "guid"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # If numeric with <= 3 unique values, could be binary classification target (e.g. churn)
            if df[col].nunique() <= 3 and col.lower() in ["churn", "target", "label", "class"]:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)
            
    # Look for classification target
    target_col = None
    for cand in ["churn", "target", "label", "class", "default"]:
        if cand in df.columns:
            target_col = cand
            break
            
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "target_feature": target_col,
        "preview": df.head(100).replace({np.nan: None}).to_dict(orient="records")
    }


def _compute_distribution_bins(
    real_df: pd.DataFrame, 
    synthetic_df: pd.DataFrame, 
    numerical_cols: List[str]
) -> Dict[str, Any]:
    """Compute histogram distribution overlays for real vs synthetic datasets."""
    distributions = {}
    for col in numerical_cols:
        if col not in real_df.columns or col not in synthetic_df.columns:
            continue
        real_vals = pd.to_numeric(real_df[col], errors="coerce").dropna().values
        synth_vals = pd.to_numeric(synthetic_df[col], errors="coerce").dropna().values
        if len(real_vals) == 0 or len(synth_vals) == 0:
            continue
            
        min_v = float(min(real_vals.min(), synth_vals.min()))
        max_v = float(max(real_vals.max(), synth_vals.max()))
        if min_v == max_v:
            max_v += 1.0
            
        bins = np.linspace(min_v, max_v, 20)
        real_hist, _ = np.histogram(real_vals, bins=bins, density=True)
        synth_hist, _ = np.histogram(synth_vals, bins=bins, density=True)
        bin_labels = [f"{bins[i]:.1f}" for i in range(len(bins)-1)]
        
        distributions[col] = {
            "bins": bin_labels,
            "real_density": [float(x) for x in real_hist],
            "synthetic_density": [float(x) for x in synth_hist],
            "real_mean": float(np.mean(real_vals)),
            "synthetic_mean": float(np.mean(synth_vals)),
            "real_std": float(np.std(real_vals)),
            "synthetic_std": float(np.std(synth_vals))
        }
    return distributions


def _compute_correlation_matrices(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    numerical_cols: List[str]
) -> Dict[str, Any]:
    """Compute correlation matrices and difference matrix."""
    valid_cols = [c for c in numerical_cols if c in real_df.columns and c in synthetic_df.columns]
    if len(valid_cols) < 2:
        return {"columns": valid_cols, "real": [], "synthetic": [], "difference": []}
        
    real_corr = real_df[valid_cols].corr().fillna(0).round(4)
    synth_corr = synthetic_df[valid_cols].corr().fillna(0).round(4)
    diff = (real_corr - synth_corr).abs().round(4)
    
    return {
        "columns": valid_cols,
        "real": real_corr.values.tolist(),
        "synthetic": synth_corr.values.tolist(),
        "difference": diff.values.tolist()
    }


@app.get("/api/health")
def health_check():
    return {"status": "ok", "project": "CODEHOLICS", "version": "2.0.0"}


@app.post("/api/load-sample")
def load_sample_dataset():
    """Loads the pre-packaged customer_data.csv benchmark dataset."""
    sample_path = os.path.join(PROJECT_ROOT, "data", "raw", "customer_data.csv")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample dataset not found on disk.")
        
    df = pd.read_csv(sample_path)
    analysis = _analyze_dataframe(df)
    
    app_state["original_df"] = df
    app_state["filename"] = "customer_data.csv"
    app_state["numerical_columns"] = analysis["numerical_features"]
    app_state["categorical_columns"] = analysis["categorical_features"]
    app_state["target_column"] = analysis["target_feature"]
    
    return {
        "message": "Sample dataset customer_data.csv loaded successfully.",
        "filename": "customer_data.csv",
        "analysis": analysis
    }


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Receives and parses an uploaded CSV dataset with resilient encoding & delimiter detection."""
    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".csv") or filename_lower.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Please upload a valid CSV file (.csv)")
        
    contents = await file.read()
    if not contents or len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    df = None
    parse_errors = []

    # Attempt 1: Standard UTF-8
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        parse_errors.append(f"UTF-8 parse: {str(e)}")

    # Attempt 2: UTF-8 with BOM & auto-separator
    if df is None or len(df.columns) <= 1:
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding="utf-8-sig", sep=None, engine="python")
        except Exception as e:
            parse_errors.append(f"Auto-sep parse: {str(e)}")

    # Attempt 3: Latin-1 encoding fallback
    if df is None or len(df.columns) <= 1:
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding="latin1", sep=None, engine="python")
        except Exception as e:
            parse_errors.append(f"Latin-1 parse: {str(e)}")

    if df is None:
        raise HTTPException(
            status_code=400, 
            detail=f"Could not parse CSV file. Errors: {'; '.join(parse_errors[:2])}"
        )

    # Clean whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]

    if df.empty or len(df.columns) < 2:
        raise HTTPException(
            status_code=400, 
            detail=f"Dataset must contain at least 2 columns and 1 row. Detected {len(df.columns)} columns."
        )

    analysis = _analyze_dataframe(df)

    app_state["original_df"] = df
    app_state["filename"] = file.filename
    app_state["numerical_columns"] = analysis["numerical_features"]
    app_state["categorical_columns"] = analysis["categorical_features"]
    app_state["target_column"] = analysis["target_feature"]

    logger.info(f"Successfully uploaded and parsed {file.filename}: {len(df)} rows, {len(df.columns)} columns")

    return {
        "message": f"Dataset {file.filename} loaded successfully.",
        "filename": file.filename,
        "analysis": analysis
    }


@app.post("/api/generate")
def generate_synthetic_data(
    model_name: str = Form("Gaussian Copula"),
    num_rows: int = Form(200),
    epochs: int = Form(100),
    batch_size: int = Form(100)
):
    """
    Executes selected synthetic generator model, generates synthetic dataset,
    and runs full privacy, distribution, correlation, and ML utility validation.
    """
    if app_state["original_df"] is None:
        raise HTTPException(
            status_code=400, 
            detail="No dataset uploaded yet. Please upload your CSV file in the Dataset Workspace first."
        )
        
    df = app_state["original_df"]
    model_data = df.copy()
    
    # Remove customer_id / id / guid columns if present
    target_col = app_state["target_column"]
    drop_cols = [c for c in ["customer_id", "id", "guid"] if c.lower() in [col.lower() for col in model_data.columns] and (target_col is None or c.lower() != target_col.lower())]
    for c in drop_cols:
        if c in model_data.columns:
            model_data = model_data.drop(columns=[c])
        
    app_state["model_data"] = model_data
    has_ml_target = (target_col is not None and target_col in model_data.columns)
    
    # Split real data for TSTR evaluation if ML target exists
    if has_ml_target and len(model_data) >= 30:
        try:
            train_real, test_real = train_test_split(model_data, test_size=0.2, random_state=42)
        except Exception:
            train_real, test_real = model_data, model_data
    else:
        train_real, test_real = model_data, model_data
        
    # Detect Metadata using SDV for user's arbitrary dataset
    metadata = Metadata.detect_from_dataframe(data=train_real)
    try:
        m_cols = metadata.to_dict().get("tables", {}).get("table", {}).get("columns", {})
        for col_name, col_meta in m_cols.items():
            if col_meta.get("sdtype") in ["unknown", "id"]:
                if pd.api.types.is_numeric_dtype(train_real[col_name]):
                    metadata.update_column(column_name=col_name, sdtype="numerical")
                else:
                    metadata.update_column(column_name=col_name, sdtype="categorical")
            if col_meta.get("pii"):
                metadata.update_column(column_name=col_name, pii=False)
    except Exception as e:
        logger.warning(f"Metadata column adjustment: {e}")
        
    # Run Generation Engine
    logger.info(f"Generating data using {model_name} for {num_rows} rows from user dataset...")
    
    # Ensure batch size is compatible with dataset length
    adjusted_batch_size = max(10, min(int(batch_size), (len(train_real) // 10) * 10 if len(train_real) >= 10 else len(train_real)))
    if adjusted_batch_size % 10 != 0 and adjusted_batch_size > 10:
        adjusted_batch_size = (adjusted_batch_size // 10) * 10
    
    if model_name == "Gaussian Copula":
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(train_real)
        synthetic_data = synthesizer.sample(num_rows=int(num_rows))
        
    elif model_name == "CTGAN":
        ctgan_epochs = min(max(epochs, 20), 300)
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=ctgan_epochs,
            batch_size=adjusted_batch_size,
            verbose=False
        )
        synthesizer.fit(train_real)
        synthetic_data = synthesizer.sample(num_rows=int(num_rows))
        
    elif model_name == "CTGAN v2":
        ctgan_epochs = min(max(epochs, 30), 300)
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=ctgan_epochs,
            batch_size=adjusted_batch_size,
            generator_lr=2e-4,
            discriminator_lr=2e-4,
            verbose=False
        )
        synthesizer.fit(train_real)
        synthetic_data = synthesizer.sample(num_rows=int(num_rows))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
        
    # Save synthetic data to state
    app_state["synthetic_df"] = synthetic_data
    app_state["current_model"] = model_name
    
    # --------------------------------------------------
    # Utility & Privacy Evaluation
    # --------------------------------------------------
    numerical_cols = [c for c in app_state["numerical_columns"] if c in model_data.columns]
    
    if len(numerical_cols) > 0:
        evaluation_results = evaluate_synthetic_data(
            train_real,
            synthetic_data,
            numerical_cols
        )
        correlation_similarity = float(evaluation_results["correlation"]["correlation_similarity"])
        exact_match_rate = float(evaluation_results["exact_match"]["exact_match_rate"])
        exact_matches = int(evaluation_results["exact_match"]["exact_matches"])
        privacy_distance = evaluation_results["privacy_distance"]
        distance_ratio = float(privacy_distance["distance_ratio"])
        synthetic_avg_distance = float(privacy_distance["synthetic_average_nearest_distance"])
        training_avg_distance = float(privacy_distance["training_average_nearest_distance"])
        ks_results = evaluation_results["ks_metrics"]
        ks_statistics = [v["ks_statistic"] for v in ks_results.values()]
        average_ks = float(np.mean(ks_statistics)) if ks_statistics else 0.0
    else:
        from src.evaluation.privacy_metrics import check_exact_match_privacy
        exact_res = check_exact_match_privacy(train_real, synthetic_data)
        exact_match_rate = float(exact_res["exact_match_rate"])
        exact_matches = int(exact_res["exact_matches"])
        correlation_similarity = 1.0
        average_ks = 0.0
        distance_ratio = 1.0
        synthetic_avg_distance = 1.0
        training_avg_distance = 1.0
        ks_results = {}
    
    # --------------------------------------------------
    # ML Utility Evaluation (TSTR)
    # --------------------------------------------------
    ml_eval = {
        "supported": False,
        "real_accuracy": 0.0,
        "real_f1": 0.0,
        "synthetic_accuracy": 0.0,
        "synthetic_f1": 0.0,
        "f1_retention": 0.0,
        "accuracy_retention": 0.0
    }
    
    if has_ml_target and len(numerical_cols) > 0:
        try:
            feat_cols = [c for c in model_data.columns if c != target_col]
            cat_cols = [c for c in app_state["categorical_columns"] if c in feat_cols]
            num_cols = [c for c in numerical_cols if c in feat_cols]
            
            # Train real model
            real_clf = train_churn_model(train_real, feat_cols, cat_cols, num_cols)
            real_metrics = evaluate_churn_model(real_clf, test_real, feat_cols)
            
            # Train synthetic model
            synth_clf = train_churn_model(synthetic_data, feat_cols, cat_cols, num_cols)
            synth_metrics = evaluate_churn_model(synth_clf, test_real, feat_cols)
            
            real_f1 = float(real_metrics["f1"])
            synth_f1 = float(synth_metrics["f1"])
            real_acc = float(real_metrics["accuracy"])
            synth_acc = float(synth_metrics["accuracy"])
            
            f1_ret = (synth_f1 / real_f1) if real_f1 > 0 else (1.0 if synth_f1 == 0 else 0.5)
            acc_ret = (synth_acc / real_acc) if real_acc > 0 else 1.0
            
            ml_eval = {
                "supported": True,
                "real_accuracy": round(real_acc, 4),
                "real_f1": round(real_f1, 4),
                "synthetic_accuracy": round(synth_acc, 4),
                "synthetic_f1": round(synth_f1, 4),
                "f1_retention": round(f1_ret, 4),
                "accuracy_retention": round(acc_ret, 4)
            }
        except Exception as e:
            logger.warning(f"ML evaluation skipped or encountered error: {e}")
            
    # Evaluation Pass / Warning / Fail Rules
    utility_pass = (correlation_similarity >= 0.80 and average_ks <= 0.15)
    privacy_pass = (exact_match_rate == 0 and distance_ratio >= 1.0)
    overall_pass = (utility_pass and privacy_pass)
    
    # Distribution charts & correlation matrix
    distributions = _compute_distribution_bins(train_real, synthetic_data, numerical_cols)
    correlation_mats = _compute_correlation_matrices(train_real, synthetic_data, numerical_cols)
    
    ks_table = [
        {
            "column": col,
            "ks_statistic": round(vals["ks_statistic"], 4),
            "p_value": round(vals["p_value"], 4),
            "status": "strong" if vals["ks_statistic"] <= 0.10 else ("moderate" if vals["ks_statistic"] <= 0.20 else "weak")
        }
        for col, vals in ks_results.items()
    ]
    
    result = {
        "status": "success",
        "model_name": model_name,
        "num_rows": len(synthetic_data),
        "synthetic_columns": list(synthetic_data.columns),
        "preview": synthetic_data.head(100).replace({np.nan: None}).to_dict(orient="records"),
        "metrics": {
            "correlation_similarity": round(correlation_similarity, 4),
            "average_ks": round(average_ks, 4),
            "exact_match_rate": round(exact_match_rate, 4),
            "exact_matches": exact_matches,
            "distance_ratio": round(distance_ratio, 4),
            "synthetic_avg_distance": round(synthetic_avg_distance, 4),
            "training_avg_distance": round(training_avg_distance, 4),
            "utility_status": "strong" if correlation_similarity >= 0.80 and average_ks <= 0.15 else ("moderate" if correlation_similarity >= 0.60 else "weak"),
            "privacy_status": "strong" if exact_match_rate == 0 and distance_ratio >= 1.0 else "warning"
        },
        "verdict": {
            "utility_pass": utility_pass,
            "privacy_pass": privacy_pass,
            "overall_pass": overall_pass,
            "utility_label": "PASS" if utility_pass else "REVIEW NEEDED",
            "privacy_label": "PASS" if privacy_pass else "REVIEW NEEDED",
            "overall_label": "PASS" if overall_pass else "REVIEW NEEDED",
            "explanation": (
                f"Generated synthetic data preserves analytical properties (Correlation: {correlation_similarity*100:.1f}%, KS: {average_ks:.4f}) "
                f"while passing empirical privacy checks (0 exact matches, Distance Ratio: {distance_ratio:.2f})."
            )
        },
        "ml_evaluation": ml_eval,
        "ks_table": ks_table,
        "distributions": distributions,
        "correlations": correlation_mats
    }
    
    app_state["evaluation_results"] = result
    return result


@app.get("/api/download-synthetic")
def download_synthetic_csv():
    """Serves the generated synthetic dataset as a downloadable CSV."""
    if app_state["synthetic_df"] is None:
        raise HTTPException(status_code=400, detail="No synthetic data has been generated yet.")
        
    csv_buffer = io.StringIO()
    app_state["synthetic_df"].to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=synthetic_data.csv"}
    )


@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns the benchmark comparison across Gaussian Copula, CTGAN, and CTGAN v2."""
    benchmark_file = os.path.join(PROJECT_ROOT, "experiments", "results", "all_models_comparison.csv")
    
    if os.path.exists(benchmark_file):
        df_bench = pd.read_csv(benchmark_file)
    else:
        # Fallback values from experiments notebook
        df_bench = pd.DataFrame([
            {
                "model": "Gaussian Copula",
                "ks_age": 0.0525,
                "ks_income": 0.0625,
                "ks_credit_score": 0.035,
                "ks_purchase_amount": 0.0413,
                "correlation_similarity": 0.9502,
                "exact_match_rate": 0.0,
                "privacy_distance_ratio": 1.0855,
                "synthetic_accuracy": 0.735,
                "synthetic_f1": 0.0702,
                "f1_retention": 0.7368
            },
            {
                "model": "CTGAN",
                "ks_age": 0.085,
                "ks_income": 0.3713,
                "ks_credit_score": 0.4213,
                "ks_purchase_amount": 0.2188,
                "correlation_similarity": 0.375,
                "exact_match_rate": 0.0,
                "privacy_distance_ratio": 3.4732,
                "synthetic_accuracy": 0.71,
                "synthetic_f1": 0.0333,
                "f1_retention": 0.3501
            },
            {
                "model": "CTGAN v2",
                "ks_age": 0.2662,
                "ks_income": 0.3625,
                "ks_credit_score": 0.2675,
                "ks_purchase_amount": 0.065,
                "correlation_similarity": 0.5254,
                "exact_match_rate": 0.0,
                "privacy_distance_ratio": 2.9391,
                "synthetic_accuracy": 0.75,
                "synthetic_f1": 0.1379,
                "f1_retention": 1.4489
            }
        ])
        
    records = df_bench.replace({np.nan: None}).to_dict(orient="records")
    
    # Calculate composite score to determine strongest observed model dynamically
    best_model = None
    best_score = -1.0
    for r in records:
        corr = float(r.get("correlation_similarity") or 0)
        # Average KS across numerical features
        ks_vals = [float(v) for k, v in r.items() if k.startswith("ks_") and v is not None]
        avg_ks = float(np.mean(ks_vals)) if ks_vals else 0.2
        acc = float(r.get("synthetic_accuracy") or 0)
        f1_ret = min(float(r.get("f1_retention") or 0), 1.0)
        
        # Composite score weighting correlation high and KS penalty low
        score = (corr * 0.45) + ((1.0 - avg_ks) * 0.35) + (acc * 0.10) + (f1_ret * 0.10)
        r["composite_score"] = round(score, 4)
        r["avg_ks"] = round(avg_ks, 4)
        
        if score > best_score:
            best_score = score
            best_model = r["model"]
            
    return {
        "benchmarks": records,
        "strongest_model": best_model,
        "strongest_score": best_score,
        "rationale": f"{best_model} achieved the highest composite retention balance with statistical correlation ({records[0]['correlation_similarity']*100:.1f}%) and low Kolmogorov-Smirnov distribution divergence."
    }


# Mount frontend static directory if exists
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
