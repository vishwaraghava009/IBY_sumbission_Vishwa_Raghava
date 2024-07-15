from transformers import pipeline

def extract_phonemes_wav2vec2(audio_path):
    pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-lv-60-espeak-cv-ft", device=0)
    
    # Load audio file
    with open(audio_path, "rb") as f:
        audio = f.read()
    
    # Get the transcription with character-level timestamps
    result = pipe(audio, return_timestamps='char')
    
    phonemes_with_timestamps = []
    for item in result["chunks"]:
        phoneme = item["text"]
        start_time = item["timestamp"][0]
        end_time = item["timestamp"][1]
        phonemes_with_timestamps.append((phoneme, start_time, end_time))
    
    return phonemes_with_timestamps

# Example usage
if __name__ == "__main__":
    audio_path = "path_to_audio.wav"
    phonemes_with_timestamps = extract_phonemes_wav2vec2(audio_path)
    print("Phonemes with timestamps:", phonemes_with_timestamps)
