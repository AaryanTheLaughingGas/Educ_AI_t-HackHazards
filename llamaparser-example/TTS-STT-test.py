from groq import Groq
import os, json

client = Groq(api_key=os.getenv("GROQ_og_API_KEY"))

def transcribe_audio(audio_path: str, prompt: str = "") -> str:
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3-turbo",
            prompt=prompt,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="en",
            temperature=0.0
        )
        return transcription.text  # or return full object for timestamps

def synthesize_speech(text: str, output_path="response.wav", voice="Fritz-PlayAI"):
    response = client.audio.speech.create(
        model="playai-tts",
        voice=voice,
        input=text,
        response_format="wav"
    )
    response.write_to_file(output_path)

import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="query.wav", duration=5, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print(f"Saved to {filename}")

# Record or load audio
audio_path = "query.wav"

# Step 1: Speech-to-Text
query_text = transcribe_audio(audio_path)
print("You said:", query_text)

# Step 2: Query Engine
response = query_engine.query(query_text)
print("LLM Response:", response)

# Step 3: Text-to-Speech
synthesize_speech(str(response), output_path="response.wav")

# Step 4: Play the result (optional)
import playsound
playsound.playsound("response.wav")