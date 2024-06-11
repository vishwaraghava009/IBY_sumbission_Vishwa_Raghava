import pyaudio
import wave
import numpy as np
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import speech_recognition as sr

def recognize_speech_from_microphone():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
        print("Got it! Now recognizing...")
        try:
            transcription = recognizer.recognize_google(audio)
            print("Transcription:", transcription)
            return transcription
        except sr.RequestError:
            print("API was unreachable or unresponsive")
        except sr.UnknownValueError:
            print("Speech was unintelligible")

def analyze_audio_tone(file_path):
    [Fs, x] = audioBasicIO.readAudioFile(file_path)
    F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
    tone_features = np.mean(F, axis=1)
    return tone_features

def record_audio(output_filename):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    audio_filename = "output.wav"
    record_audio(audio_filename)
    transcription = recognize_speech_from_microphone()
    tone_features = analyze_audio_tone(audio_filename)
    print("Tone features:", tone_features)
