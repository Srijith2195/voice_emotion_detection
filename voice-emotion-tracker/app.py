from flask import Flask, render_template, request
import os
import joblib
from audio_utils.record import record_voice
from audio_utils.preprocess import extract_features
from audio_utils.waveform import plot_waveform

# 👇 Tell Flask where to find templates
app = Flask(__name__, template_folder="app/templates")

# 👇 Load your trained model
model = joblib.load(os.path.join("model", "emotion_model.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    emotion = None
    features = None

    if request.method == "POST":
        if "record" in request.form:
            record_voice("input.wav")
            plot_waveform("input.wav")  # ✅ Waveform visualization
            features = extract_features("input.wav")

        elif "file" in request.files:
            f = request.files["file"]
            if f and f.filename.endswith(".wav"):
                f.save("input.wav")
                plot_waveform("input.wav")  # ✅ Waveform visualization
                features = extract_features("input.wav")
            else:
                emotion = "Only .wav files are supported."

        if features is not None:
            try:
                prediction = model.predict([features])
                emotion = prediction[0]
            except Exception as e:
                emotion = f"Prediction failed: {str(e)}"

    return render_template("index.html", emotion=emotion)

if __name__ == "__main__":
    app.run(debug=True)
