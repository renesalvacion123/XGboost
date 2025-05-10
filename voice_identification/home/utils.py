# utils.py

import librosa
import numpy as np
import os

# Helper function to check if a file exists
def file_exists(file_path):
    return os.path.exists(file_path)


# Helper function to check if a file is a valid audio file (e.g., .wav, .mp3)
def is_audio_file(filename):
    valid_extensions = ['.wav', '.mp3', '.flac', '.webm']  # Add other extensions if needed
    return any(filename.endswith(ext) for ext in valid_extensions)

# Function to extract features from audio using librosa
def extract_features(audio_path):
    # Load the audio file using librosa
    y, sr = librosa.load(audio_path, sr=None)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)

    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    # Extract Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Combine all features into one feature vector
    features = np.hstack([mfcc_mean, mfcc_var, spectral_contrast_mean, zcr_mean])

    return features


def classify_voice(file_path, model):
    features = extract_features(file_path)
    if features is None:
        return "Error: Failed to extract features"
    
    # Reshape features to match the input shape expected by the model (if necessary)
    features = features.reshape(1, -1)
    prediction = model.predict(features)  # Predict using the loaded model
    return "Fake" if prediction[0] == 1 else "Real"