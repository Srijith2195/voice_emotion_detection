from flask import Flask, render_template, request, send_file, send_from_directory, redirect, url_for
import os
import joblib
from audio_utils.record import record_voice
from audio_utils.preprocess import extract_features
from audio_utils.waveform import plot_waveform
import speech_recognition as sr
import json
from datetime import datetime
from scipy.io import wavfile
import shutil
import uuid

# 👇 Tell Flask where to find templates
app = Flask(__name__, template_folder="app/templates")

# 👇 Load your trained model
model = joblib.load(os.path.join("model", "emotion_model.pkl"))
HISTORY_FILE = "history.json"
AUDIO_DIR = os.path.join("static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

def get_audio_properties(filepath):
    try:
        rate, data = wavfile.read(filepath)
        duration = data.shape[0] / rate
        channels = 1 if len(data.shape) == 1 else data.shape[1]
        return {
            'sample_rate': rate,
            'channels': channels,
            'duration': round(duration, 2),
            'format': 'WAV'
        }
    except Exception:
        return None

@app.route("/", methods=["GET", "POST"])
@app.route("/upload", methods=["GET", "POST"])
def index():
    emotion = None
    features = None
    transcription = None
    audio_available = False
    audio_props = None
    audio_filename = None
    if request.method == "POST":
        unique_id = uuid.uuid4().hex
        audio_filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}.wav"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        if "record" in request.form:
            record_voice("input.wav")
            plot_waveform("input.wav")
            features = extract_features("input.wav")
            if os.path.exists("input.wav"):
                shutil.copy("input.wav", audio_path)
        elif "file" in request.files:
            f = request.files["file"]
            if f and f.filename.endswith(".wav"):
                f.save("input.wav")
                plot_waveform("input.wav")
                features = extract_features("input.wav")
                shutil.copy("input.wav", audio_path)
            else:
                emotion = "Only .wav files are supported."
        # Speech-to-text transcription
        if os.path.exists("input.wav"):
            audio_available = True
            audio_props = get_audio_properties("input.wav")
            recognizer = sr.Recognizer()
            with sr.AudioFile("input.wav") as source:
                audio_data = recognizer.record(source)
                try:
                    transcription = recognizer.recognize_google(audio_data, language='en-US')
                except sr.UnknownValueError:
                    transcription = "Could not understand audio."
                except sr.RequestError as e:
                    transcription = f"Speech recognition error: {e}"
        if features is not None:
            try:
                prediction = model.predict([features])
                emotion = prediction[0]
                # Save to history
                record = {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": audio_filename,
                    "emotion": emotion,
                    "transcription": transcription
                }
                if os.path.exists(HISTORY_FILE):
                    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                        history = json.load(f)
                else:
                    history = []
                history.insert(0, record)  # newest first
                with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
            except Exception as e:
                emotion = f"Prediction failed: {str(e)}"
    return render_template("index.html", emotion=emotion, transcription=transcription, audio_available=audio_available, audio_props=audio_props, active_page="home", audio_filename=audio_filename)

@app.route("/audio/<filename>")
def audio(filename):
    # Serve the latest input.wav for playback
    return send_from_directory(AUDIO_DIR, filename)

@app.route("/history")
def history():
    # Load history from file
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    return render_template("history.html", active_page="history", history=history)

@app.route("/analytics")
def analytics():
    # Load history and count emotions
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    emotion_counts = {}
    for record in history:
        emotion = record.get("emotion", "Unknown")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    return render_template("analytics.html", active_page="analytics", emotion_counts=emotion_counts)

@app.route("/settings")
def settings():
    # Placeholder for settings
    return render_template("settings.html", active_page="settings")

@app.route("/delete_history/<int:index>", methods=["POST"])
def delete_history(index):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        if 0 <= index < len(history):
            # Remove audio file
            audio_file = history[index].get("filename")
            if audio_file:
                audio_path = os.path.join(AUDIO_DIR, audio_file)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            # Remove entry
            history.pop(index)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
    return redirect(url_for("history"))

if __name__ == "__main__":
    app.run(debug=True)