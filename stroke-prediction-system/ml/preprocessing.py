"""
Data preprocessing and feature engineering for stroke prediction.
"""
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class StrokeDataPreprocessor:
    """Handles all data preprocessing for stroke prediction."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.bmi_median = None

    def _engineer_features(self, df):
        """Create model-ready features without fitting/scaling."""
        df = df.copy()

        # Handle missing BMI values with training median
        if self.bmi_median is None:
            self.bmi_median = df['bmi'].median()
        df['bmi'] = df['bmi'].fillna(self.bmi_median)

        # Create clinically meaningful bins
        df['age_group'] = pd.cut(df['age'], bins=[0, 50, 80, 120], labels=[0, 1, 2]).astype(int)
        df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
        df['glucose_category'] = pd.cut(
            df['avg_glucose_level'], bins=[0, 100, 126, 300], labels=[0, 1, 2]
        ).astype(int)

        # Encode categorical variables
        df['gender_encoded'] = df['gender'].map({
            'male': 1, 'Male': 1,
            'female': 0, 'Female': 0,
            'other': 2, 'Other': 2,
        }).fillna(2)

        df['ever_married_encoded'] = df['ever_married'].map({
            'yes': 1, 'Yes': 1,
            'no': 0, 'No': 0,
        }).fillna(0)

        df = pd.concat([df, pd.get_dummies(df['work_type'], prefix='work')], axis=1)
        df = pd.concat([df, pd.get_dummies(df['smoking_status'], prefix='smoking')], axis=1)

        df['urban_residence'] = df['Residence_type'].map({
            'urban': 1, 'Urban': 1,
            'rural': 0, 'Rural': 0,
        }).fillna(0)

        feature_cols = [
            'age', 'hypertension', 'heart_disease', 'bmi', 'avg_glucose_level',
            'age_group', 'bmi_category', 'glucose_category',
            'gender_encoded', 'ever_married_encoded', 'urban_residence',
        ]
        feature_cols.extend([col for col in df.columns if col.startswith('work_')])
        feature_cols.extend([col for col in df.columns if col.startswith('smoking_')])

        return df[feature_cols].copy()

    def preprocess(self, df, fit=True):
        """
        Preprocess the stroke dataset.

        Args:
            df: Raw dataframe
            fit: Whether to fit preprocessing artifacts (True for training, False for inference)

        Returns:
            Processed feature matrix and target (if available)
        """
        df = df.copy()

        X = self._engineer_features(df)

        if fit:
            self.feature_names = X.columns.tolist()
        elif self.feature_names is not None:
            # Ensure exact feature schema used during training
            X = X.reindex(columns=self.feature_names, fill_value=0)

        numerical_cols = [col for col in ['age', 'bmi', 'avg_glucose_level'] if col in X.columns]
        if fit:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])

        y = df['stroke'] if 'stroke' in df.columns else None
        return X, y

    def save(self, path='models/preprocessor.pkl'):
        """Save preprocessor state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'bmi_median': self.bmi_median,
            },
            path,
        )

    def load(self, path='models/preprocessor.pkl'):
        """Load preprocessor state."""
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.bmi_median = state['bmi_median']


def prepare_data(data_path='data/stroke-data.csv', test_size=0.2, random_state=42):
    """
    Load and prepare data for training.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    df = pd.read_csv(data_path)

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    y = df['stroke']
    X_raw = df.drop(columns=['stroke'])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Re-attach target for preprocessing helper output compatibility
    train_df = X_train_raw.copy()
    train_df['stroke'] = y_train.values
    test_df = X_test_raw.copy()
    test_df['stroke'] = y_test.values

    preprocessor = StrokeDataPreprocessor()
    X_train, _ = preprocessor.preprocess(train_df, fit=True)
    X_test, _ = preprocessor.preprocess(test_df, fit=False)

    return X_train, X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True), preprocessor
