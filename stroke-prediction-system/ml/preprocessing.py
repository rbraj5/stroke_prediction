"""
Preprocessing contract for the stroke prediction pipeline.

This module keeps the feature-building rules in one place so that training,
saved artifacts, and inference all use the same assumptions. In practice,
this file acts as the contract between raw patient input and the model-ready
feature set.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# This version tags the preprocessing contract rather than the model itself.
# If feature logic changes later, the saved preprocessing artifact should be
# treated as a different contract.
PREPROCESSOR_VERSION = "v1"

# Training data contains the target column. Inference payloads from the API or
# dashboard do not. Keeping the name central avoids hard-coded references in
# multiple methods.
TARGET_COLUMN = "stroke"

# This is the public raw-input contract shared across the training pipeline,
# API schema, and dashboard form. Any field added or removed here should be
# reviewed as an interface change, not just a local code edit.
RAW_INPUT_FIELDS: List[str] = [
    "age",
    "gender",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

# These fields are required to build the current baseline features. If any of
# them are missing, preprocessing should fail clearly instead of inferring
# intent from incomplete input.
REQUIRED_RAW_INPUT_FIELDS: List[str] = [
    "age",
    "gender",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "smoking_status",
]

# BMI is optional because the pipeline has an explicit fallback for missing
# values. Listing optional fields separately makes that behaviour visible to
# reviewers and easier to validate in code.
OPTIONAL_RAW_INPUT_FIELDS: List[str] = [
    "bmi",
]

# These field groups are defined once so validation and transformation logic
# can stay consistent and readable.
NUMERIC_RAW_FIELDS: List[str] = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
]

CATEGORICAL_RAW_FIELDS: List[str] = [
    "gender",
    "smoking_status",
]

# Allowed categories are kept explicit so that input validation and encoding
# rules do not drift apart over time.
ALLOWED_GENDER_VALUES = {"Male", "Female", "Other"}
ALLOWED_SMOKING_STATUS_VALUES = {
    "never smoked",
    "formerly smoked",
    "smokes",
    "Unknown",
}

# The current baseline keeps age as a continuous feature. That works well for
# both logistic regression and tree-based models, and it avoids introducing
# arbitrary age cutoffs into the preprocessing layer.
ENGINEERED_FEATURES: List[str] = []

# These continuous fields are scaled after feature assembly. Keeping the list
# explicit makes the fitted preprocessing state easier to review and update.
SCALED_FEATURES: List[str] = [
    "age",
    "avg_glucose_level",
    "bmi",
]

# This is the frozen feature schema consumed by the trained model. The order is
# intentional: training and inference must produce columns in the same order.
# In a team setting, a change here should be treated as a model-contract change.
SELECTED_FEATURES: List[str] = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi",
    "gender_other",
    "smoking_status_formerly smoked",
    "smoking_status_smokes",
    "smoking_status_Unknown",
]

# These source columns exist in the dataset but are not part of the current
# baseline model. Keeping the rationale next to the contract makes feature
# scope easier to explain during review and easier to revisit later.
EXCLUDED_FEATURES: Dict[str, str] = {
    "id": "Identifier only; not clinically meaningful for prediction.",
    "ever_married": "Excluded in the notebook as low predictive value.",
    "work_type": "Excluded in the notebook as low predictive value.",
    "Residence_type": "Excluded in the notebook as low predictive value.",
}

# These notes document the reasoning behind the current preprocessing design.
# They are also reused by downstream metadata consumers such as the API and
# dashboard, so the explanation stays consistent across the project.
PREPROCESSING_DECISIONS: Dict[str, str] = {
    "bmi_imputation": (
        "BMI missing values are filled with the training-set median. "
        "This keeps more rows available for modelling and avoids relying on a "
        "separate BMI prediction step."
    ),
    "age_handling": (
        "Age is kept as a continuous input in the current baseline. "
        "This preserves more information for both logistic regression and tree-based models."
    ),
    "gender_inclusivity": (
        "The rare 'Other' gender category is retained rather than dropped so the "
        "pipeline does not silently exclude minority cases."
    ),
}

# Presentation labels are defined here so training metadata, API responses,
# and UI code can describe the same features consistently.
DISPLAY_FEATURE_LABELS: Dict[str, str] = {
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart disease",
    "avg_glucose_level": "Average glucose level",
    "bmi": "BMI",
    "gender_other": "Gender = Other",
    "smoking_status_formerly smoked": "Formerly smoked",
    "smoking_status_smokes": "Smokes",
    "smoking_status_Unknown": "Smoking status unknown",
}


class StrokeDataPreprocessor:
    """Convert raw patient data into a stable model-ready feature matrix."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.bmi_median: float | None = None
        self.selected_features = SELECTED_FEATURES.copy()
        self.excluded_features = EXCLUDED_FEATURES.copy()
        self.preprocessing_decisions = PREPROCESSING_DECISIONS.copy()
        self.preprocessor_version = PREPROCESSOR_VERSION

    def _ensure_optional_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add optional columns that may be absent in inference payloads.

        The current pipeline allows BMI to be missing because it has a defined
        imputation path. Adding the column here keeps later transformation code
        simpler and avoids conditional checks in multiple places.
        """
        df = df.copy()

        for column in OPTIONAL_RAW_INPUT_FIELDS:
            if column not in df.columns:
                df[column] = pd.NA

        return df

    def _normalize_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise raw string inputs before validation and encoding.

        Different entry points may provide values with extra whitespace or
        inconsistent casing. Normalising first prevents valid inputs from
        failing validation for formatting-only reasons.
        """
        df = df.copy()

        if "gender" in df.columns:
            df["gender"] = df["gender"].apply(
                lambda value: value if pd.isna(value) else str(value).strip().title()
            )

        if "smoking_status" in df.columns:
            smoking_status_map = {
                "never smoked": "never smoked",
                "formerly smoked": "formerly smoked",
                "smokes": "smokes",
                "unknown": "Unknown",
            }
            df["smoking_status"] = df["smoking_status"].apply(
                lambda value: value
                if pd.isna(value)
                else smoking_status_map.get(str(value).strip().lower(), str(value).strip())
            )

        return df

    def _validate_input(self, df: pd.DataFrame, expect_target: bool) -> None:
        """
        Validate the raw input contract before feature generation starts.

        Training and inference use the same raw feature definitions, but only
        training data is expected to carry the target column.
        """
        missing_required = [column for column in REQUIRED_RAW_INPUT_FIELDS if column not in df.columns]
        if missing_required:
            raise ValueError(
                f"Missing required input columns: {missing_required}. "
                f"Expected columns: {RAW_INPUT_FIELDS}"
            )

        if expect_target and TARGET_COLUMN not in df.columns:
            raise ValueError(f"Training data must include target column '{TARGET_COLUMN}'.")

        if not expect_target and TARGET_COLUMN in df.columns:
            raise ValueError(
                f"Transform input should not include target column '{TARGET_COLUMN}'."
            )

        # Extra columns are allowed at this layer. Feature selection happens
        # explicitly in the transformation step, so unknown source columns do
        # not automatically flow into the model.

        numeric_required_fields = ["age", "hypertension", "heart_disease", "avg_glucose_level"]
        for column in numeric_required_fields:
            numeric_series = pd.to_numeric(df[column], errors="coerce")
            if numeric_series.isna().any():
                raise ValueError(
                    f"Column '{column}' contains values that cannot be converted to numeric."
                )

        # Required categorical fields must be present and populated. Allowing
        # nulls here would silently turn incomplete records into all-zero
        # baseline dummies during one-hot encoding.
        categorical_required_fields = ["gender", "smoking_status"]
        for column in categorical_required_fields:
            if df[column].isna().any():
                raise ValueError(
                    f"Column '{column}' contains missing values, but this field is required."
                )

        for column in ["hypertension", "heart_disease"]:
            binary_values = pd.to_numeric(df[column], errors="coerce").dropna().unique()
            invalid_binary_values = set(binary_values) - {0, 1}
            if invalid_binary_values:
                raise ValueError(
                    f"Column '{column}' must contain only 0/1 values. "
                    f"Found: {sorted(invalid_binary_values)}"
                )

        if "bmi" in df.columns:
            bmi_non_null = df["bmi"].dropna()
            if not bmi_non_null.empty:
                bmi_numeric = pd.to_numeric(bmi_non_null, errors="coerce")
                if bmi_numeric.isna().any():
                    raise ValueError("Column 'bmi' contains non-numeric values.")

        invalid_gender_values = set(df["gender"].dropna().unique()) - ALLOWED_GENDER_VALUES
        if invalid_gender_values:
            raise ValueError(
                f"Invalid gender values found: {sorted(invalid_gender_values)}. "
                f"Allowed values: {sorted(ALLOWED_GENDER_VALUES)}"
            )

        invalid_smoking_values = set(df["smoking_status"].dropna().unique()) - ALLOWED_SMOKING_STATUS_VALUES
        if invalid_smoking_values:
            raise ValueError(
                f"Invalid smoking_status values found: {sorted(invalid_smoking_values)}. "
                f"Allowed values: {sorted(ALLOWED_SMOKING_STATUS_VALUES)}"
            )

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert normalised raw inputs into the fixed feature schema expected by
        the model.

        The current baseline keeps the transformation logic explicit rather than
        using a generic auto-encoding step. That makes the final feature set
        easier to review and easier to keep stable across training and inference.
        """
        if self.bmi_median is None:
            bmi_non_null = pd.to_numeric(df["bmi"], errors="coerce").dropna()
            if bmi_non_null.empty:
                raise ValueError(
                    "BMI median cannot be learned because all training BMI values are missing."
                )
            self.bmi_median = float(bmi_non_null.median())

        df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce").fillna(self.bmi_median)

        # Numeric fields are coerced explicitly so that downstream scaling and
        # model input assembly do not depend on implicit pandas conversions.
        numeric_df = pd.DataFrame(index=df.index)
        numeric_df["age"] = pd.to_numeric(df["age"], errors="coerce")
        numeric_df["hypertension"] = pd.to_numeric(df["hypertension"], errors="coerce").fillna(0).astype(int)
        numeric_df["heart_disease"] = pd.to_numeric(df["heart_disease"], errors="coerce").fillna(0).astype(int)
        numeric_df["avg_glucose_level"] = pd.to_numeric(df["avg_glucose_level"], errors="coerce")
        numeric_df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

        # One-hot encoding is kept selective on purpose. The model uses only the
        # non-baseline indicators that are part of the frozen feature schema.
        gender_dummies = pd.get_dummies(df["gender"], prefix="gender")
        smoking_dummies = pd.get_dummies(df["smoking_status"], prefix="smoking_status")

        feature_df = pd.DataFrame(index=df.index)
        feature_df["age"] = numeric_df["age"]
        feature_df["hypertension"] = numeric_df["hypertension"]
        feature_df["heart_disease"] = numeric_df["heart_disease"]
        feature_df["avg_glucose_level"] = numeric_df["avg_glucose_level"]
        feature_df["bmi"] = numeric_df["bmi"]
        feature_df["gender_other"] = gender_dummies.get("gender_Other", pd.Series(0, index=df.index))
        feature_df["smoking_status_formerly smoked"] = smoking_dummies.get(
            "smoking_status_formerly smoked", pd.Series(0, index=df.index)
        )
        feature_df["smoking_status_smokes"] = smoking_dummies.get(
            "smoking_status_smokes", pd.Series(0, index=df.index)
        )
        feature_df["smoking_status_Unknown"] = smoking_dummies.get(
            "smoking_status_Unknown", pd.Series(0, index=df.index)
        )

        # The model should only see the approved feature schema and in the
        # approved order. Reindexing here makes that contract explicit.
        return feature_df.reindex(columns=self.selected_features, fill_value=0)

    def fit(self, df: pd.DataFrame) -> "StrokeDataPreprocessor":
        """
        Learn preprocessing state from training data.

        This step captures anything that must be reused later at inference time,
        such as the BMI imputation value, fitted scaler, and final feature order.
        """
        df = df.copy()
        df = self._ensure_optional_columns(df)
        df = self._normalize_inputs(df)
        self._validate_input(df, expect_target=True)

        X = self._engineer_features(df)
        X = X.apply(pd.to_numeric, errors="coerce")
        self.feature_names = X.columns.tolist()

        scaled_columns = [column for column in SCALED_FEATURES if column in X.columns]
        X[scaled_columns] = X[scaled_columns].fillna(0)
        self.scaler.fit(X[scaled_columns])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted preprocessing state to raw feature inputs.

        This method is used after fit() for validation, test, and inference
        data. It does not require the target column.
        """
        if not self.feature_names:
            raise ValueError("Preprocessor has not been fitted. Call fit() before transform().")

        df = df.copy()
        df = self._ensure_optional_columns(df)
        df = self._normalize_inputs(df)
        self._validate_input(df, expect_target=False)

        X = self._engineer_features(df)
        X = X.reindex(columns=self.feature_names, fill_value=0)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        scaled_columns = [column for column in SCALED_FEATURES if column in X.columns]
        X[scaled_columns] = self.scaler.transform(X[scaled_columns])
        return X

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the preprocessor on training data and return transformed features
        with the aligned target.
        """
        self.fit(df)
        X = self.transform(df.drop(columns=[TARGET_COLUMN]))
        y = df[TARGET_COLUMN].reset_index(drop=True)
        return X, y

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series | None]:
        """
        Backward-compatible wrapper for existing call sites.

        Training calls return X and y. Inference calls return X and None.
        """
        if fit:
            return self.fit_transform(df)

        inference_df = df.drop(columns=[TARGET_COLUMN]) if TARGET_COLUMN in df.columns else df
        return self.transform(inference_df), None

    def get_metadata(self) -> Dict[str, object]:
        """
        Return the preprocessing contract in a format that downstream layers can
        inspect and present.

        This keeps training, API, and UI code aligned on the same view of the
        feature pipeline without duplicating assumptions across files.
        """
        return {
            "preprocessor_version": self.preprocessor_version,
            "target_column": TARGET_COLUMN,
            "raw_input_fields": RAW_INPUT_FIELDS,
            "required_raw_input_fields": REQUIRED_RAW_INPUT_FIELDS,
            "optional_raw_input_fields": OPTIONAL_RAW_INPUT_FIELDS,
            "numeric_raw_fields": NUMERIC_RAW_FIELDS,
            "categorical_raw_fields": CATEGORICAL_RAW_FIELDS,
            "allowed_gender_values": sorted(ALLOWED_GENDER_VALUES),
            "allowed_smoking_status_values": sorted(ALLOWED_SMOKING_STATUS_VALUES),
            "engineered_features": ENGINEERED_FEATURES,
            "selected_features": self.selected_features,
            "excluded_features": self.excluded_features,
            "preprocessing_decisions": self.preprocessing_decisions,
            "display_feature_labels": DISPLAY_FEATURE_LABELS,
            "bmi_median": self.bmi_median,
            "scaled_features": SCALED_FEATURES,
        }

    def save(self, path: str | Path = "models/preprocessor.pkl") -> None:
        """
        Persist the fitted preprocessing state required for inference.

        The saved artifact should contain only the state needed to reproduce the
        training-time feature contract at inference time.
        """
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        os.makedirs(path.parent, exist_ok=True)
        joblib.dump(
            {
                "preprocessor_version": self.preprocessor_version,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "bmi_median": self.bmi_median,
                "selected_features": self.selected_features,
                "excluded_features": self.excluded_features,
                "preprocessing_decisions": self.preprocessing_decisions,
            },
            path,
        )

    def load(self, path: str | Path = "models/preprocessor.pkl") -> None:
        """
        Load a fitted preprocessing artifact for inference or evaluation.

        Loading restores the exact feature schema and scaling state used when
        the training artifacts were created.
        """
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        state = joblib.load(path)
        self.preprocessor_version = state.get("preprocessor_version", "unknown")
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self.bmi_median = state["bmi_median"]
        self.selected_features = state.get("selected_features", SELECTED_FEATURES.copy())
        self.excluded_features = state.get("excluded_features", EXCLUDED_FEATURES.copy())
        self.preprocessing_decisions = state.get("preprocessing_decisions", PREPROCESSING_DECISIONS.copy())


def prepare_data(
    data_path: str | Path = "data/stroke-data.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StrokeDataPreprocessor]:
    """
    Load the dataset, split it into train/test partitions, and apply the shared
    preprocessing contract.

    The split happens before fitting the preprocessor so that imputation and
    scaling parameters are learned from training data only.
    """
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    df = pd.read_csv(data_path)

    # The source dataset includes an identifier column that is not part of the
    # modeling contract.
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    y = df[TARGET_COLUMN]
    X_raw = df.drop(columns=[TARGET_COLUMN])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = X_train_raw.copy()
    train_df[TARGET_COLUMN] = y_train.values

    preprocessor = StrokeDataPreprocessor()
    X_train, y_train_processed = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(X_test_raw)

    return (
        X_train,
        X_test,
        y_train_processed.reset_index(drop=True),
        y_test.reset_index(drop=True),
        preprocessor,
    )
