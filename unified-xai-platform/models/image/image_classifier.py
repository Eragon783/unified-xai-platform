"""
Image Classifier Module
Handles lung cancer detection using DenseNet121 with transfer learning.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from pathlib import Path


class ImageClassifier:
    """Classifier for chest X-ray lung cancer detection using DenseNet121."""

    CLASS_NAMES = ['benign', 'malignant']

    def __init__(self, model_path: str = None, use_pretrained: bool = True):
        """
        Initialize the image classifier.

        Args:
            model_path: Path to a saved model. If None, uses pretrained DenseNet121.
            use_pretrained: If True and no model_path, creates DenseNet121 with ImageNet weights.
        """
        self.model = None
        self.model_path = model_path
        self.use_pretrained = use_pretrained
        self._load_model()

    def _load_model(self):
        """Load or create the model."""
        if self.model_path and Path(self.model_path).exists():
            # Load saved model
            self.model = tf.keras.models.load_model(str(self.model_path))
        elif self.use_pretrained:
            # Create DenseNet121 with ImageNet weights for transfer learning
            self.model = self._create_densenet_model()
        else:
            raise ValueError("No model path provided and use_pretrained is False")

    def _create_densenet_model(self) -> Model:
        """
        Create DenseNet121 model configured for binary classification.
        Uses ImageNet pretrained weights with custom classification head.

        Returns:
            Configured Keras Model
        """
        # Load DenseNet121 without top layers
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )

        # Freeze base model layers (optional, can be fine-tuned later)
        base_model.trainable = False

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        # Binary classification output
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=outputs)

        return model

    def predict(self, image_array: np.ndarray) -> tuple:
        """
        Predict whether X-ray shows benign or malignant condition.

        Args:
            image_array: Preprocessed image array.
                        Shape should be (H, W, 3) or (batch, H, W, 3)
                        Values should be normalized to [0, 1]

        Returns:
            Tuple of (class_label, prediction_probabilities)
        """
        # Ensure batch dimension
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)

        # Apply DenseNet preprocessing
        processed = tf.keras.applications.densenet.preprocess_input(
            image_array * 255.0  # preprocess_input expects [0, 255]
        )

        # Get prediction
        prediction = self.model.predict(processed, verbose=0)
        class_label = np.argmax(prediction[0])

        return class_label, prediction

    def predict_proba(self, image_array: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (for XAI methods).

        Args:
            image_array: Preprocessed image array(s) normalized to [0, 1]

        Returns:
            Prediction probabilities array
        """
        # Ensure batch dimension
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)

        # Apply DenseNet preprocessing
        processed = tf.keras.applications.densenet.preprocess_input(
            image_array * 255.0
        )

        return self.model.predict(processed, verbose=0)

    def get_class_name(self, class_label: int) -> str:
        """
        Get the class name for a given label.

        Args:
            class_label: Integer class label (0 or 1)

        Returns:
            Class name string ('benign' or 'malignant')
        """
        return self.CLASS_NAMES[class_label]

    def get_model(self) -> Model:
        """
        Get the underlying Keras model (for XAI methods).

        Returns:
            The Keras model
        """
        return self.model

    def get_last_conv_layer_name(self) -> str:
        """
        Get the name of the last convolutional layer for Grad-CAM.

        Returns:
            Layer name string
        """
        # DenseNet121's last conv layer
        return 'conv5_block16_concat'

    def save_model(self, path: str):
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        self.model.save(path)
