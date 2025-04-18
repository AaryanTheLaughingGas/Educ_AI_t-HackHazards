import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 7  # Duration

print("Recording...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
write("test_mic.wav", fs, recording)
print("Saved recording to test_mic.wav")