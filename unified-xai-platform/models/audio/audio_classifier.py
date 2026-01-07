"""
Audio Classifier Module
Handles deepfake audio detection using pre-trained models.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import os

# Use tf_keras (Keras 2) for legacy model compatibility
try:
    import tf_keras as keras
except ImportError:
    # Fallback to standard keras if tf_keras not available
    keras = tf.keras


class AudioClassifier:
    """Classifier for deepfake audio detection."""

    CLASS_NAMES = ['real', 'fake']

    def __init__(self, model_path: str = None):
        """
        Initialize the audio classifier.

        Args:
            model_path: Path to the saved model. If None, uses default MobileNet model.
        """
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the pre-trained model."""
        if self.model_path is None:
            # Default path relative to this file
            base_dir = Path(__file__).parent.parent.parent
            self.model_path = base_dir / 'assets' / 'saved_models' / 'audio' / 'mobilenet'

        model_path = Path(self.model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        model_path_str = str(model_path)

        # Use tf_keras (Keras 2 API) for loading legacy models
        self.model = keras.models.load_model(model_path_str)

    def predict(self, image_array: np.ndarray) -> tuple:
        """
        Predict whether audio (as spectrogram) is real or fake.

        Args:
            image_array: Preprocessed spectrogram image array.
                        Shape should be (H, W, 3) or (batch, H, W, 3)
                        Values should be normalized to [0, 1]

        Returns:
            Tuple of (class_label, prediction_probabilities)
        """
        # Ensure batch dimension
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)

        # Ensure values are normalized
        if image_array.max() > 1.0:
            image_array = image_array / 255.0

        # Get prediction
        prediction = self.model.predict(image_array, verbose=0)
        class_label = np.argmax(prediction[0])

        return class_label, prediction

    def predict_proba(self, image_array: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (for XAI methods).

        Args:
            image_array: Preprocessed spectrogram image array(s)

        Returns:
            Prediction probabilities array
        """
        # Ensure batch dimension
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)

        # Ensure values are normalized
        if image_array.max() > 1.0:
            image_array = image_array / 255.0

        return self.model.predict(image_array, verbose=0)

    def get_class_name(self, class_label: int) -> str:
        """
        Get the class name for a given label.

        Args:
            class_label: Integer class label (0 or 1)

        Returns:
            Class name string ('real' or 'fake')
        """
        return self.CLASS_NAMES[class_label]

    def get_model(self) -> tf.keras.Model:
        """
        Get the underlying Keras model (for XAI methods).

        Returns:
            The loaded Keras model
        """
        return self.model
