"""
Model training and evaluation for stroke prediction.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    recall_score, precision_score, f1_score, roc_curve
)
import joblib
import json
import os
from datetime import datetime


class StrokePredictor:
    """Stroke prediction model wrapper."""
    
    def __init__(self, model_type='balanced_rf'):
        """
        Initialize model.
        
        Args:
            model_type: One of 'balanced_rf', 'random_forest', 'logistic', 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = self._create_model(model_type)
        self.feature_importance = None
        self.training_metrics = {}
        
    def _create_model(self, model_type):
        """Create model based on type."""
        if model_type == 'balanced_rf':
            return BalancedRandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic':
            return LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train the model and calculate metrics."""
        print(f"Training {self.model_type} model...")
        
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                X_train.columns,
                self.model.feature_importances_
            ))
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.training_metrics = {
            'model_type': self.model_type,
            'accuracy': float(self.model.score(X_test, y_test)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'training_date': datetime.now().isoformat(),
            'n_training_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Model Performance - {self.model_type}")
        print(f"{'='*50}")
        print(f"Accuracy:  {self.training_metrics['accuracy']:.4f}")
        print(f"Precision: {self.training_metrics['precision']:.4f}")
        print(f"Recall:    {self.training_metrics['recall']:.4f}")
        print(f"F1-Score:  {self.training_metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {self.training_metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TN: {self.training_metrics['confusion_matrix'][0][0]}, "
              f"FP: {self.training_metrics['confusion_matrix'][0][1]}")
        print(f"FN: {self.training_metrics['confusion_matrix'][1][0]}, "
              f"TP: {self.training_metrics['confusion_matrix'][1][1]}")
        print(f"{'='*50}\n")
        
        return self.training_metrics
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, path='models/stroke_model.pkl'):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path='models/stroke_model.pkl'):
        """Load model from disk."""
        state = joblib.load(path)
        self.model = state['model']
        self.model_type = state['model_type']
        self.feature_importance = state.get('feature_importance')
        self.training_metrics = state.get('training_metrics', {})
        print(f"Model loaded from {path}")


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance."""
    models = ['balanced_rf', 'random_forest', 'logistic', 'gradient_boosting']
    results = {}
    
    for model_type in models:
        predictor = StrokePredictor(model_type)
        metrics = predictor.train(X_train, y_train, X_test, y_test)
        results[model_type] = metrics
        
        # Save each model
        predictor.save(f'models/stroke_{model_type}.pkl')
    
    # Save comparison results
    os.makedirs('models', exist_ok=True)
    with open('models/model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best model based on recall (most important for medical applications)
    best_model = max(results.items(), key=lambda x: x[1]['recall'])
    print(f"\nBest model (by recall): {best_model[0]} - Recall: {best_model[1]['recall']:.4f}")
    
    return results, best_model[0]


if __name__ == '__main__':
    from preprocessing import prepare_data
    
    # Prepare data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.pkl')
    
    # Train and compare models
    results, best_model_name = train_and_compare_models(X_train, X_test, y_train, y_test)
    
    # Train final best model
    print(f"\nTraining final {best_model_name} model...")
    final_model = StrokePredictor(best_model_name)
    final_model.train(X_train, y_train, X_test, y_test)
    final_model.save('models/stroke_model_production.pkl')
    
    print("\n✅ Training complete! Models saved to models/ directory")
