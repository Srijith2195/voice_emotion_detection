import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

def plot_waveform(file_path, save_path="static/waveform.png"):
    try:
        rate, data = wavfile.read(file_path)

        # Handle stereo audio by selecting one channel
        if len(data.shape) > 1:
            data = data[:, 0]

        plt.figure(figsize=(8, 3))
        plt.plot(data, color="#7c3aed")
        plt.title("Voice Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()

        # Ensure static directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Waveform plot failed: {e}")
