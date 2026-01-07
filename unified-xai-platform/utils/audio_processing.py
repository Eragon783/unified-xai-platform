"""
Audio Processing Utilities
Handles audio file loading and spectrogram conversion for deepfake detection.
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import os
import tempfile


def load_audio(file_path: str) -> tuple:
    """
    Load an audio file using librosa.

    Args:
        file_path: Path to the audio file (.wav)

    Returns:
        Tuple of (audio_samples, sample_rate)
    """
    y, sr = librosa.load(file_path)
    return y, sr


def create_spectrogram(audio_path: str, output_path: str, target_size: tuple = (224, 224)) -> str:
    """
    Convert audio file to mel-spectrogram image.

    Args:
        audio_path: Path to the input audio file
        output_path: Path to save the spectrogram image
        target_size: Target image size (width, height)

    Returns:
        Path to the saved spectrogram image
    """
    # Load audio
    y, sr = librosa.load(audio_path)

    # Create mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Create figure without axes for clean image
    fig, ax = plt.subplots(figsize=(3, 3))
    librosa.display.specshow(log_mel_spec, sr=sr, ax=ax)
    ax.axis('off')

    # Save the spectrogram
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # Resize to target size
    img = Image.open(output_path)
    img = img.resize(target_size)
    img.save(output_path)

    return output_path


def preprocess_spectrogram_for_model(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess spectrogram image for model input.

    Args:
        image_path: Path to the spectrogram image
        target_size: Expected input size for the model

    Returns:
        Preprocessed numpy array ready for model prediction
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        file_path: Path to the audio file

    Returns:
        Duration in seconds
    """
    y, sr = librosa.load(file_path)
    return librosa.get_duration(y=y, sr=sr)


def create_spectrogram_from_upload(uploaded_file, temp_dir: str) -> tuple:
    """
    Create spectrogram from a Streamlit uploaded file.
    Matches the original app.py behavior.

    Args:
        uploaded_file: Streamlit UploadedFile object
        temp_dir: Directory to save temporary files

    Returns:
        Tuple of (spectrogram_image_data, spectrogram_path)
    """
    # Save uploaded file temporarily
    audio_path = os.path.join(temp_dir, uploaded_file.name)
    with open(audio_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Create spectrogram matching original implementation
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_path)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    # Save spectrogram
    spec_path = os.path.join(temp_dir, 'melspectrogram.png')
    plt.savefig(spec_path)
    plt.close(fig)

    # Load as image data for model (224x224)
    image_data = load_img(spec_path, target_size=(224, 224))

    return image_data, spec_path


def spectrogram_to_array(image_data, normalize: bool = True) -> np.ndarray:
    """
    Convert spectrogram image to numpy array for model input.

    Args:
        image_data: PIL Image or image data from load_img
        normalize: Whether to normalize to [0, 1]

    Returns:
        Numpy array ready for model prediction
    """
    img_array = np.array(image_data)
    if normalize:
        img_array = img_array / 255.0
    return img_array
