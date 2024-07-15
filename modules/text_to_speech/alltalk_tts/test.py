import html
import json
import random
import subprocess
import time
import os
import requests
import threading
import signal
import sys
import atexit
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import re
import numpy as np
import soundfile as sf
import uuid
import logging

try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ModuleNotFoundError:
    # Inform the user about the missing module and suggest next steps
    print(f"[{params['branding']}]\033[91mWarning\033[0m Could not find the TTS module. Make sure to install the requirements for the {params['branding']} extension.")
    print(f"[{params['branding']}]\033[91mWarning\033[0m Please use the ATSetup utility or check the Github installation instructions.")
    # Re-raise the ModuleNotFoundError to stop the program and print the traceback
    raise 


# Initialize the TTS model (Assuming TTS is installed and working locally)
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
with open(this_dir / "system" / "config" / "languages.json", encoding="utf8") as f:
    languages = json.load(f)
# Load the config file
def load_config(file_path):
    with open(file_path, "r") as config_file:
        config = json.load(config_file)
    return config

# Path to the config file
this_dir = Path(__file__).parent.resolve()
config_file_path = this_dir / "confignew.json"
params = load_config(config_file_path)

# Update config function (as in the original script)
def update_config(config_file_path, update_config_path, downgrade_config_path):
    # Similar to the original function
    pass

# Function to get available voices
def get_available_voices():
    return sorted([voice.name for voice in Path(f"{this_dir}/voices").glob("*.wav")])

# Function to delete old files
def delete_old_files(folder_path, days_to_keep):
    current_time = datetime.now()
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            age = current_time - file_creation_time
            if age > timedelta(days=days_to_keep):
                os.remove(file_path)

# Initial setup for TTS
def before_audio_generation(string, params):
    string = html.unescape(string) or random_sentence()
    if string == "":
        return "*Empty string*"
    return string

# Function to combine audio files
def combine(audio_files, output_folder, state):
    audio = np.array([])
    for audio_file in audio_files:
        audio_data, sample_rate = sf.read(audio_file)
        if audio.size == 0:
            audio = audio_data
        else:
            audio = np.concatenate((audio, audio_data))
    output_file_path = os.path.join(output_folder, f"TTSOUT_{int(time.time())}_combined.wav")
    sf.write(output_file_path, audio, samplerate=sample_rate)
    for audio_file in audio_files:
        os.remove(audio_file)
    return output_file_path

# Generate random sentence
def random_sentence():
    with open(this_dir / "system" / "config" / "harvard_sentences.txt") as f:
        return random.choice(list(f))

# Process text (as in the original script)
def process_text(text):
    text = html.unescape(text)
    text = re.sub(r"\.{3,}", ".", text)
    combined_pattern = r'(\*[^*"]+\*|"[^"*]+")'
    ordered_parts = []
    start = 0
    for match in re.finditer(combined_pattern, text):
        if start < match.start():
            ambiguous_text = text[start : match.start()].strip()
            if ambiguous_text:
                ordered_parts.append(("ambiguous", ambiguous_text))
        matched_text = match.group(0)
        if matched_text.startswith("*") and matched_text.endswith("*"):
            ordered_parts.append(("narrator", matched_text.strip("*").strip()))
        elif matched_text.startswith('"') and matched_text.endswith('"'):
            ordered_parts.append(("character", matched_text.strip('"').strip()))
        else:
            if "*" in matched_text:
                ordered_parts.append(("narrator", matched_text.strip("*").strip('"')))
            else:
                ordered_parts.append(("character", matched_text.strip('"').strip("*")))
        start = match.end()
    if start < len(text):
        ambiguous_text = text[start:].strip()
        if ambiguous_text:
            ordered_parts.append(("ambiguous", ambiguous_text))
    return ordered_parts

# Main function to generate TTS output
def output_modifier(string, state):
    if not params["activate"]:
        return string
    cleaned_string = before_audio_generation(string, params)
    if cleaned_string is None:
        return
    language_code = languages.get(params["language"])
    temperature = params["local_temperature"]
    repetition_penalty = params["local_repetition_penalty"]
    audio_files = []
    if params["narrator_enabled"]:
        processed_parts = process_text(cleaned_string)
        audio_files_all_paragraphs = []
        for part_type, part in processed_parts:
            if len(part.strip()) <= 3:
                continue
            voice_to_use = params["narrator_voice"] if part_type == "narrator" else params["voice"]
            cleaned_part = re.sub(r"([!?.\u3002\uFF1F\uFF01\uFF0C])\1+", r"\1", part)
            cleaned_part = re.sub(r"\u2026{1,2}", ". ", cleaned_part)
            cleaned_part = re.sub(r'[^a-zA-Z0-9\s.,;:!?\-\'"$]', '', cleaned_part)
            cleaned_part = re.sub(r"\n+", " ", cleaned_part)
            output_filename = get_output_filename(state)
            tts.tts_to_file(text=cleaned_part, file_path=output_filename, speaker=voice_to_use)
            audio_files_all_paragraphs.append(output_filename)
        final_output_file = combine(audio_files_all_paragraphs, params["output_folder_wav"], state)
    else:
        cleaned_part = html.unescape(cleaned_string)
        cleaned_part = re.sub(r"([!?.\u3002\uFF1F\uFF01\uFF0C])\1+", r"\1", cleaned_part)
        cleaned_part = re.sub(r"\u2026{1,2}", ". ", cleaned_part)
        cleaned_part = re.sub(r'[^a-zA-Z0-9\s.,;:!?\-\'"$]', '', cleaned_part)
        cleaned_part = re.sub(r"\n+", " ", cleaned_part)
        output_file = Path(f'{params["output_folder_wav"]}/TTSOUT_{int(time.time())}.wav')
        tts.tts_to_file(text=cleaned_part, file_path=output_file, speaker=params["voice"])
        final_output_file = output_file
    return final_output_file

def get_output_filename(state):
    return Path(f'{params["output_folder_wav"]}/TTSOUT_{str(uuid.uuid4())[:8]}.wav').as_posix()

# Example usage
if __name__ == "__main__":
    state = {}
    text_to_speak = "Hello, this is a test of the TTS system."
    output_file = output_modifier(text_to_speak, state)
    print(f"Generated audio file: {output_file}")
