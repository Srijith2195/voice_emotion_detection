import sounddevice as sd
from scipy.io.wavfile import write

def record_voice(filename="input.wav", duration=10, fs=44100):  # ⏱️ Updated to 10 seconds
    try:
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"{i}: {dev['name']}")

        # Use the default microphone device
        device_info = sd.query_devices(kind='input')
        print(f"\nUsing default input device: {device_info['name']}")

        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=device_info['index'])
        sd.wait()
        write(filename, fs, audio)
        print("Recording saved as", filename)

    except Exception as e:
        print("⚠️ Error recording audio:", e)
        print("Try closing other apps using the mic, or manually specify a device.")
