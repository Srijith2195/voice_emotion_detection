import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/ravdess/"
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file):
    y, sr = librosa.load(file, duration=3, offset=0.5)
    return np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

X, y = [], []
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav'):
            emotion = emotions[file.split("-")[2]]
            features = extract_features(os.path.join(root, file))
            X.append(features)
            y.append(emotion)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "model/emotion_model.pkl")
print("Model trained and saved.")
