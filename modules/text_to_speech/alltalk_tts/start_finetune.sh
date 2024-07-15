#!/bin/bash
export TRAINER_TELEMETRY=0
export LD_LIBRARY_PATH=/home/iby_vishwa/Documents/SillyTavern-extras/modules/text_to_speech/alltalk_tts/alltalk_environment/env/lib/python3.11/site-packages/nvidia/cublas/lib:/home/iby_vishwa/Documents/SillyTavern-extras/modules/text_to_speech/alltalk_tts/alltalk_environment/env/lib/python3.11/site-packages/nvidia/cudnn/lib
source "/home/iby_vishwa/Documents/SillyTavern-extras/modules/text_to_speech/alltalk_tts/alltalk_environment/conda/etc/profile.d/conda.sh"
conda activate "/home/iby_vishwa/Documents/SillyTavern-extras/modules/text_to_speech/alltalk_tts/alltalk_environment/env"
python finetune.py
