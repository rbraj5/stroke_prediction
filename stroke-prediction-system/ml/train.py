"""
Training and evaluation for the stroke prediction portfolio project.

The goal of this module is not just to fit a model. It also packages the
current notebook-backed baseline into reusable artifacts and rich metadata so
future model upgrades can reuse the same API and dashboard surfaces.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml.preprocessing import StrokeDataPreprocessor, prepare_data

MODELS_DIR = PROJECT_ROOT / "models"
ACTIVE_MODEL_NAME = "logistic"
ACTIVE_MODEL_VERSION = "baseline-v1"
SCREENING_THRESHOLD = 0.30

MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "logistic": {
        "label": "Logistic Regression",
        "factory": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "selection_note": (
            "Chosen as the active baseline because it preserves strong recall while remaining the most interpretable "
            "option for a screening-style use case."
        ),
        "rejection_note": "Not rejected; this is the current active baseline.",
        "explanation_method": "Signed logistic coefficients translated into plain-language factor summaries.",
    },
    "balanced_rf": {
        "label": "Balanced Random Forest",
        "factory": lambda: BalancedRandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "selection_note": (
            "High recall candidate, but the current portfolio keeps it as an alternative rather than the active model "
            "because it is harder to explain to non-technical healthcare stakeholders."
        ),
        "rejection_note": "Rejected for the active baseline because interpretability mattered more than a small recall edge.",
        "explanation_method": "Global tree feature importance only.",
    },
    "random_forest": {
        "label": "Random Forest",
        "factory": lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "selection_note": "Competitive metrics, but the current baseline prefers a model with cleaner decision transparency.",
        "rejection_note": "Rejected for the active baseline because it adds complexity without a clear interview-ready advantage.",
        "explanation_method": "Global tree feature importance only.",
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "factory": lambda: GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42,
        ),
        "selection_note": "Useful comparison point for accuracy-heavy behavior.",
        "rejection_note": "Rejected because it missed too many stroke-risk cases for a recall-focused screening problem.",
        "explanation_method": "Global tree feature importance only.",
    },
}


class StrokePredictor:
    """Wrap a classifier with thresholding and model metadata for inference."""

    def __init__(self, model_type: str = ACTIVE_MODEL_NAME, threshold: float = SCREENING_THRESHOLD) -> None:
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}")
        self.model_type = model_type
        self.model_label = str(MODEL_REGISTRY[model_type]["label"])
        self.threshold = threshold
        self.model = self._create_model(model_type)
        self.feature_importance: Dict[str, float] | None = None
        self.feature_coefficients: Dict[str, float] | None = None
        self.training_metrics: Dict[str, object] = {}
        self.explanation_method = str(MODEL_REGISTRY[model_type]["explanation_method"])
        self.model_version = ACTIVE_MODEL_VERSION

    def _create_model(self, model_type: str):
        return MODEL_REGISTRY[model_type]["factory"]()

    def _derive_explanation_metadata(self, X_train: pd.DataFrame) -> None:
        """Capture model-specific explanation details without changing API contracts."""
        if hasattr(self.model, "coef_"):
            coefficients = self.model.coef_[0]
            self.feature_coefficients = {feature: float(weight) for feature, weight in zip(X_train.columns, coefficients)}
            self.feature_importance = {
                feature: float(abs(weight)) for feature, weight in zip(X_train.columns, coefficients)
            }
        elif hasattr(self.model, "feature_importances_"):
            self.feature_importance = {
                feature: float(weight) for feature, weight in zip(X_train.columns, self.model.feature_importances_)
            }
            self.feature_coefficients = None
        else:
            self.feature_importance = None
            self.feature_coefficients = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
        """Fit the model and calculate screening-focused metrics using the configured threshold."""
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self._derive_explanation_metadata(X_train)

        y_prob = self.predict_proba(X_test)[:, 1]
        y_pred = self.apply_threshold(y_prob)

        self.training_metrics = {
            "model_type": self.model_type,
            "model_label": self.model_label,
            "threshold": self.threshold,
            "accuracy": float((y_pred == y_test).mean()),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "training_date": datetime.now().isoformat(),
            "n_training_samples": len(X_train),
            "n_test_samples": len(X_test),
            "selection_note": MODEL_REGISTRY[self.model_type]["selection_note"],
            "rejection_note": MODEL_REGISTRY[self.model_type]["rejection_note"],
            "explanation_method": self.explanation_method,
        }

        print(f"\n{'=' * 50}")
        print(f"Model Performance - {self.model_type}")
        print(f"{'=' * 50}")
        print(f"Accuracy:  {self.training_metrics['accuracy']:.4f}")
        print(f"Precision: {self.training_metrics['precision']:.4f}")
        print(f"Recall:    {self.training_metrics['recall']:.4f}")
        print(f"F1-Score:  {self.training_metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {self.training_metrics['roc_auc']:.4f}")
        print(f"Threshold: {self.threshold:.2f}")
        print(f"\nConfusion Matrix:")
        print(f"TN: {self.training_metrics['confusion_matrix'][0][0]}, FP: {self.training_metrics['confusion_matrix'][0][1]}")
        print(f"FN: {self.training_metrics['confusion_matrix'][1][0]}, TP: {self.training_metrics['confusion_matrix'][1][1]}")
        print(f"{'=' * 50}\n")

        return self.training_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return thresholded predictions so inference matches evaluation."""
        probabilities = self.predict_proba(X)[:, 1]
        return self.apply_threshold(probabilities)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def apply_threshold(self, probabilities: np.ndarray) -> np.ndarray:
        """Use an explicit screening threshold instead of the library default 0.5."""
        return (probabilities >= self.threshold).astype(int)

    def build_explanation_summary(self, X: pd.DataFrame, top_n: int = 3) -> List[Dict[str, object]]:
        """Return a stable explanation shape that future models can also fill."""
        if self.feature_coefficients:
            scored = []
            for feature, coefficient in self.feature_coefficients.items():
                feature_value = float(X.iloc[0][feature])
                contribution = coefficient * feature_value
                if contribution == 0:
                    continue
                scored.append(
                    {
                        "feature": feature,
                        "contribution": float(contribution),
                        "direction": "increases risk" if contribution > 0 else "decreases risk",
                    }
                )
            return sorted(scored, key=lambda item: abs(item["contribution"]), reverse=True)[:top_n]

        if self.feature_importance:
            ranked = sorted(self.feature_importance.items(), key=lambda item: item[1], reverse=True)[:top_n]
            return [
                {
                    "feature": feature,
                    "contribution": float(weight),
                    "direction": "globally influential",
                }
                for feature, weight in ranked
            ]

        return []

    def save(self, path: str | Path = "models/stroke_model.pkl") -> None:
        """Persist the active model with enough metadata for stable inference."""
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        os.makedirs(path.parent, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "model_type": self.model_type,
                "model_label": self.model_label,
                "threshold": self.threshold,
                "feature_importance": self.feature_importance,
                "feature_coefficients": self.feature_coefficients,
                "training_metrics": self.training_metrics,
                "explanation_method": self.explanation_method,
                "model_version": self.model_version,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str | Path = "models/stroke_model.pkl") -> None:
        """Load a persisted model artifact used by the API."""
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        state = joblib.load(path)
        self.model = state["model"]
        self.model_type = state["model_type"]
        self.model_label = state.get("model_label", str(MODEL_REGISTRY[self.model_type]["label"]))
        self.threshold = state.get("threshold", SCREENING_THRESHOLD)
        self.feature_importance = state.get("feature_importance")
        self.feature_coefficients = state.get("feature_coefficients")
        self.training_metrics = state.get("training_metrics", {})
        self.explanation_method = state.get("explanation_method", str(MODEL_REGISTRY[self.model_type]["explanation_method"]))
        self.model_version = state.get("model_version", ACTIVE_MODEL_VERSION)
        print(f"Model loaded from {path}")


def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, StrokePredictor]]:
    """Train candidate models into a shared metrics shape for the dashboard and API."""
    results: Dict[str, Dict[str, object]] = {}
    predictors: Dict[str, StrokePredictor] = {}

    for model_type in MODEL_REGISTRY:
        predictor = StrokePredictor(model_type=model_type, threshold=SCREENING_THRESHOLD)
        metrics = predictor.train(X_train, y_train, X_test, y_test)
        results[model_type] = metrics
        predictors[model_type] = predictor
        predictor.save(MODELS_DIR / f"stroke_{model_type}.pkl")

    return results, predictors


def build_model_metadata(
    comparison_results: Dict[str, Dict[str, object]],
    active_predictor: StrokePredictor,
    preprocessor: StrokeDataPreprocessor,
) -> Dict[str, object]:
    """Build the stable metadata contract consumed by the API and dashboard."""
    preprocessor_metadata = preprocessor.get_metadata()

    candidate_models = []
    for model_type, result in comparison_results.items():
        candidate_models.append(
            {
                "model_type": model_type,
                "label": result["model_label"],
                "status": "active" if model_type == ACTIVE_MODEL_NAME else "rejected",
                "selection_rationale": result["selection_note"],
                "rejection_rationale": result["rejection_note"],
                "metrics": {
                    "accuracy": result["accuracy"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1_score": result["f1_score"],
                    "roc_auc": result["roc_auc"],
                },
                "confusion_matrix": result["confusion_matrix"],
            }
        )

    active_metrics = active_predictor.training_metrics
    return {
        "artifact_version": 1,
        "active_model": {
            "model_type": active_predictor.model_type,
            "label": active_predictor.model_label,
            "version": active_predictor.model_version,
            "training_date": active_metrics.get("training_date"),
            "threshold": active_predictor.threshold,
            "threshold_rationale": (
                "The decision threshold is set below 0.50 to prioritize recall and reduce false negatives, "
                "which fits a screening use case better than the default classifier threshold."
            ),
            "why_this_version": (
                "Logistic regression is the current active baseline because it keeps recall high while making the model "
                "easier to justify to clinicians and interviewers than less transparent alternatives."
            ),
            "primary_objective": "High recall screening support with fewer false negatives.",
            "use_case": "Stroke risk screening support, not diagnosis.",
            "metrics": {
                "accuracy": active_metrics.get("accuracy"),
                "precision": active_metrics.get("precision"),
                "recall": active_metrics.get("recall"),
                "f1_score": active_metrics.get("f1_score"),
                "roc_auc": active_metrics.get("roc_auc"),
            },
            "confusion_matrix": active_metrics.get("confusion_matrix"),
            "explanation_method": active_predictor.explanation_method,
        },
        "candidate_models": candidate_models,
        "feature_decisions": {
            "preprocessor_version": preprocessor_metadata["preprocessor_version"],
            "target_column": preprocessor_metadata["target_column"],
            "required_raw_input_fields": preprocessor_metadata["required_raw_input_fields"],
            "optional_raw_input_fields": preprocessor_metadata["optional_raw_input_fields"],
            "numeric_raw_fields": preprocessor_metadata["numeric_raw_fields"],
            "categorical_raw_fields": preprocessor_metadata["categorical_raw_fields"],
            "allowed_gender_values": preprocessor_metadata["allowed_gender_values"],
            "allowed_smoking_status_values": preprocessor_metadata["allowed_smoking_status_values"],
            "engineered_features": preprocessor_metadata["engineered_features"],
            "selected_features": preprocessor_metadata["selected_features"],
            "selected_feature_count": len(preprocessor_metadata["selected_features"]),
            "excluded_features": preprocessor_metadata["excluded_features"],
            "display_feature_labels": preprocessor_metadata["display_feature_labels"],
            "preprocessing_decisions": preprocessor_metadata["preprocessing_decisions"],
            "raw_input_fields": preprocessor_metadata["raw_input_fields"],
            "scaled_features": preprocessor_metadata["scaled_features"],
        },
        "workflow_summary": {
            "steps": [
                "Raw data intake",
                "Contract-first preprocessing with validation, imputation, encoding, and scaling",
                "Candidate-model comparison",
                "Threshold tuning for screening recall",
                "Saved model and metadata artifacts",
                "FastAPI inference layer",
                "Streamlit dashboard",
            ],
            "engineering_signals": [
                "Reusable preprocessing shared by training and inference",
                "Strict fit/transform separation for preprocessing state",
                "Explicit artifact metadata for model choice and thresholding",
                "API-backed prediction flow",
                "Automated tests for preprocessing and API behavior",
                "Extensible model architecture for future experiments",
            ],
        },
        "limitations": [
            "Portfolio and educational system; not clinically validated for care decisions.",
            "Current explanation output is a baseline model summary, not a certified clinical interpretation tool.",
            "The architecture is designed to support future experiments such as ensembles, calibration, and richer explainability.",
        ],
        "next_steps": [
            "Add calibrated probability analysis.",
            "Experiment with ensembles and compare them using the same metadata contract.",
            "Introduce stronger explainability for non-linear models.",
            "Add fairness and monitoring checks as the project matures.",
        ],
    }


def save_metadata(metadata: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(PROJECT_ROOT / "data/stroke-data.csv")

    preprocessor.save(MODELS_DIR / "preprocessor.pkl")

    comparison_results, predictors = compare_models(X_train, X_test, y_train, y_test)
    save_metadata(comparison_results, MODELS_DIR / "model_comparison.json")

    active_predictor = predictors[ACTIVE_MODEL_NAME]
    active_predictor.save(MODELS_DIR / "stroke_model_production.pkl")

    model_metadata = build_model_metadata(comparison_results, active_predictor, preprocessor)
    save_metadata(model_metadata, MODELS_DIR / "model_metadata.json")

    print("\nTraining complete. Artifacts and metadata saved to models/ directory")





