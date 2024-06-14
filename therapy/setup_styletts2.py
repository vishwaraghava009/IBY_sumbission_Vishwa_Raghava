import os
import subprocess
import sys

def run_command(command, shell=False):
    result = subprocess.run(command, shell=shell, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(result.stderr)
    else:
        print(result.stdout)

# Clone the repository
if not os.path.exists('StyleTTS2'):
    run_command(['git', 'clone', 'https://github.com/yl4579/StyleTTS2.git'])

# Change to the StyleTTS2 directory
os.chdir('StyleTTS2')

# Install dependencies
dependencies = [
    'SoundFile',
    'torchaudio',
    'munch',
    'torch',
    'pydub',
    'pyyaml',
    'librosa',
    'nltk',
    'matplotlib',
    'accelerate',
    'transformers',
    'phonemizer',
    'einops',
    'einops-exts',
    'tqdm',
    'typing-extensions'
]

run_command(['pip', 'install'] + dependencies)

# Install monotonic_align separately and handle potential build issues
try:
    run_command(['pip', 'install', 'git+https://github.com/resemble-ai/monotonic_align.git'])
except Exception as e:
    print("Failed to install monotonic_align. Please install it manually.")
    print(e)

# Download model files
if not os.path.exists('Models'):
    run_command(['git', 'clone', 'https://huggingface.co/yl4579/StyleTTS2-LJSpeech'])
    run_command(['mv', 'StyleTTS2-LJSpeech/Models', '.'], shell=True)

# For Windows, skip the espeak-ng installation part
if sys.platform.startswith('win'):
    print("Please install espeak-ng manually on Windows.")
else:
    # Install espeak-ng for non-Windows systems
    run_command(['sudo', 'apt-get', 'install', 'espeak-ng'], shell=True)

print("Setup completed successfully.")
