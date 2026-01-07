"""
Tabular Data Processing Utilities
Handles CSV file loading and preprocessing for fraud detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os


def load_csv_from_upload(uploaded_file, temp_dir: str = None) -> tuple:
    """
    Load a CSV file from Streamlit upload.

    Args:
        uploaded_file: Streamlit uploaded file object
        temp_dir: Directory to save temporary files

    Returns:
        Tuple of (DataFrame, file_path)
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    # Save the uploaded file temporarily
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Load the CSV
    df = pd.read_csv(file_path)

    return df, file_path


def validate_fraud_csv(df: pd.DataFrame) -> dict:
    """
    Validate that the CSV has the expected structure for fraud detection.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with validation results
    """
    expected_features = [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]

    result = {
        'valid': True,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'missing_features': [],
        'available_features': [],
        'has_class_column': 'Class' in df.columns,
        'warnings': []
    }

    # Check for expected features
    for feature in expected_features:
        if feature in df.columns:
            result['available_features'].append(feature)
        else:
            result['missing_features'].append(feature)

    # Determine if we have enough features
    if len(result['available_features']) < 5:
        result['valid'] = False
        result['warnings'].append(
            f"Too few matching features. Found only {len(result['available_features'])} of {len(expected_features)} expected features."
        )

    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 5:
        result['valid'] = False
        result['warnings'].append("Dataset should contain mostly numeric columns.")

    return result


def get_sample_for_analysis(df: pd.DataFrame, row_index: int = 0) -> pd.DataFrame:
    """
    Get a single sample from the DataFrame for analysis.

    Args:
        df: Input DataFrame
        row_index: Index of the row to analyze

    Returns:
        Single-row DataFrame
    """
    if row_index >= len(df):
        row_index = 0

    return df.iloc[[row_index]]


def prepare_features_for_display(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Prepare feature data for display in Streamlit.

    Args:
        df: Input DataFrame
        feature_names: List of feature names to display

    Returns:
        Formatted DataFrame for display
    """
    available = [f for f in feature_names if f in df.columns]
    display_df = df[available].copy()

    # Round numeric values for better display
    for col in display_df.select_dtypes(include=[np.number]).columns:
        display_df[col] = display_df[col].round(4)

    return display_df


def generate_sample_fraud_csv() -> pd.DataFrame:
    """
    Generate a sample fraud detection CSV for demo purposes.

    Returns:
        DataFrame with sample fraud detection data
    """
    np.random.seed(42)
    n_samples = 10

    # Generate sample data
    data = {
        'Time': np.random.randint(0, 172800, n_samples),
    }

    # Add V1-V28 features (PCA transformed)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)

    # Add Amount
    data['Amount'] = np.abs(np.random.randn(n_samples)) * 100

    # Add Class (mostly legitimate with a few frauds)
    data['Class'] = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]

    return pd.DataFrame(data)
