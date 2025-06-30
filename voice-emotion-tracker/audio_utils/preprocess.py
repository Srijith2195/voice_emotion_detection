import numpy as np
import librosa

def extract_features(file_name):
    try:
        y, sr = librosa.load(file_name, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc
    except Exception as e:
        print("Error:", e)
        return None
