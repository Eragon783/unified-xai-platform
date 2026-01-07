"""
Fraud Detection Classifier Module
Handles credit card fraud detection using XGBoost.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class FraudClassifier:
    """Classifier for credit card fraud detection."""

    CLASS_NAMES = ['legitimate', 'fraud']

    # Expected features for the fraud detection model
    # Based on typical credit card fraud datasets (anonymized PCA features + Amount + Time)
    EXPECTED_FEATURES = [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]

    def __init__(self, model_path: str = None):
        """
        Initialize the fraud classifier.

        Args:
            model_path: Path to the saved model. If None, creates a demo model.
        """
        self.model = None
        self.scaler = None
        self.model_path = model_path
        self.feature_names = None
        self._load_or_create_model()

    def _load_or_create_model(self):
        """Load a pre-trained model or create a demo model."""
        if self.model_path is not None:
            model_path = Path(self.model_path)
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.scaler = saved_data.get('scaler')
                    self.feature_names = saved_data.get('feature_names', self.EXPECTED_FEATURES)
                return

        # Create a demo RandomForest model trained on synthetic data
        # This allows the platform to work without requiring a real dataset
        self._create_demo_model()

    def _create_demo_model(self):
        """Create a demo model with synthetic data for demonstration purposes."""
        np.random.seed(42)

        # Generate synthetic fraud detection data
        n_samples = 1000
        n_features = len(self.EXPECTED_FEATURES)

        # Generate features (simulating PCA-transformed data)
        X = np.random.randn(n_samples, n_features)

        # Add some patterns for fraud detection
        # Fraudulent transactions tend to have certain feature patterns
        fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        X[fraud_indices, 0] += 2  # V1 tends to be higher for fraud
        X[fraud_indices, 3] -= 1.5  # V4 tends to be lower for fraud
        X[fraud_indices, -1] = np.abs(X[fraud_indices, -1]) * 500  # Higher amounts

        # Create labels (0 = legitimate, 1 = fraud)
        y = np.zeros(n_samples)
        y[fraud_indices] = 1

        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train a RandomForest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.feature_names = self.EXPECTED_FEATURES

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input DataFrame for prediction.

        Args:
            df: Input DataFrame with transaction features

        Returns:
            Preprocessed numpy array ready for prediction
        """
        # Select only the expected features that exist in the dataframe
        available_features = [f for f in self.feature_names if f in df.columns]

        if len(available_features) == 0:
            raise ValueError(
                f"No matching features found. Expected features like: {self.feature_names[:5]}"
            )

        X = df[available_features].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Scale if scaler is available
        if self.scaler is not None:
            # If we have fewer features than expected, we need to handle this
            if X.shape[1] == len(self.feature_names):
                X = self.scaler.transform(X)
            else:
                # Partial scaling - create a new scaler for available features
                temp_scaler = StandardScaler()
                X = temp_scaler.fit_transform(X)

        return X

    def predict(self, X: np.ndarray) -> tuple:
        """
        Predict whether a transaction is fraudulent.

        Args:
            X: Preprocessed feature array

        Returns:
            Tuple of (class_label, prediction_probabilities)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        prediction = self.model.predict_proba(X)
        class_label = np.argmax(prediction[0])

        return class_label, prediction

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Preprocessed feature array

        Returns:
            Prediction probabilities array
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> dict:
        """
        Get feature importance from the model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        return {}

    def get_class_name(self, class_label: int) -> str:
        """
        Get the class name for a given label.

        Args:
            class_label: Integer class label (0 or 1)

        Returns:
            Class name string ('legitimate' or 'fraud')
        """
        return self.CLASS_NAMES[class_label]

    def get_model(self):
        """
        Get the underlying model (for XAI methods).

        Returns:
            The trained model
        """
        return self.model

    def get_feature_names(self) -> list:
        """
        Get the feature names used by the model.

        Returns:
            List of feature names
        """
        return self.feature_names

    def save_model(self, path: str):
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
