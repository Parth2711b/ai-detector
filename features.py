import librosa
import numpy as np

def extract_features(file_path):
    """
    Extract baseline audio features from an audio file.
    Returns a fixed-length numpy array.
    """

    # Load audio (keep original sample rate)
    y, sr = librosa.load(file_path, sr=None)

    # Handle very short or broken audio
    if y is None or len(y) == 0:
        raise ValueError("Empty audio file")

    # MFCC features (most important baseline)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    # Take mean across time axis
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean
