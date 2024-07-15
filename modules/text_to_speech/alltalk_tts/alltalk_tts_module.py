import requests
import os
import json
from modules.text_to_speech.alltalk_tts.phoneme_extraction import extract_phonemes_wav2vec2
from modules.text_to_speech.alltalk_tts.phoneme_to_morph import phonemes_to_morph_indices, phoneme_to_morph_map

def generate_audio(text, voice, output_file_name, json_output_path, language, text_filtering='standard', narrator_enabled=False, narrator_voice_gen='female_01.wav', text_not_inside='character', autoplay=False, autoplay_volume=0.8, output_file_timestamp=False):
    url = "http://127.0.0.1:7851/api/tts-generate"
    payload = {
        "text_input": text,
        "text_filtering": text_filtering,
        "character_voice_gen": voice,
        "narrator_enabled": str(narrator_enabled).lower(),
        "narrator_voice_gen": narrator_voice_gen,
        "text_not_inside": text_not_inside,
        "language": language,
        "output_file_name": output_file_name,
        "output_file_timestamp": str(output_file_timestamp).lower(),
        "autoplay": str(autoplay).lower(),
        "autoplay_volume": autoplay_volume
    }

    print("Payload being sent to the API:", payload)

    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        result = response.json()
        if result.get("status") == "generate-success":
            audio_path = result.get("output_file_path")
            audio_url = result.get("output_file_url")

            print("Generated audio file path:", audio_path)
            print("Generated audio file URL:", audio_url)

            try:
                phonemes_with_timestamps = extract_phonemes_wav2vec2(audio_path)
                print("Phonemes with timestamps extracted successfully:")
                print(phonemes_with_timestamps)

                frame_duration = 1 / 25  # Duration of each frame in seconds (adjusted for animation)
                morph_indices_with_timestamps = []

                for phoneme, start_time, end_time in phonemes_with_timestamps:
                    morph = phoneme_to_morph_map.get(phoneme, 'mouth_aaa_index')
                    start_frame = int(start_time / frame_duration)
                    end_frame = int(end_time / frame_duration)
                    
                    for frame in range(start_frame, end_frame + 1):
                        morph_indices_with_timestamps.append({
                            "morph": morph,
                            "start": frame * frame_duration,
                            "end": (frame + 1) * frame_duration
                        })

                print("Morph indices with timestamps:", morph_indices_with_timestamps)

                # Save the morph indices with timestamps to a JSON file
                with open(json_output_path, 'w') as json_file:
                    json.dump(morph_indices_with_timestamps, json_file, indent=4)

                print(f"Morph indices with timestamps saved to {json_output_path}")

                return audio_path, audio_url, morph_indices_with_timestamps
            except Exception as e:
                print("Error in phoneme extraction:", str(e))
                return audio_path, audio_url, None
        else:
            print("Audio generation failed:", result)
            return None, None, None
    else:
        print("Request failed with status code:", response.status_code, response.text)
        return None, None, None

# # Example usage
# if __name__ == "__main__":
#     text = "Heyy, What's up buddy? I know you are tired with all these presentations! Don't worry. I'm not gonna take too much time."
#     voice = "male_03.wav"
#     output_file_name = "myoutputfile"
#     json_output_path = "/home/iby_vishwa/Documents/SillyTavern-extras/modules/text_to_speech/alltalk_tts/outputs/morph_indices.json"

#     file_path, file_url, morph_indices_with_timestamps = generate_audio(text, voice, output_file_name, json_output_path)
#     print("Generated audio file path:", file_path)
#     print("Generated audio file URL:", file_url)
#     print("Morph indices with timestamps:", morph_indices_with_timestamps)
