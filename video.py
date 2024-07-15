import os
import time
import threading
from PIL import Image
import numpy as np
import torch
import cv2
from talkinghead.tha3.poser.modes.load_poser import load_poser
from talkinghead.tha3.poser.poser import Poser
from talkinghead.tha3.util import (torch_linear_to_srgb, resize_PIL_image,
                       extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image)
from talkinghead.tha3.app.postprocessor import Postprocessor
from talkinghead.tha3.app.util import posedict_keys, posedict_key_to_index, load_emotion_presets, posedict_to_pose, to_talkinghead_image, RunningAverage

# Import the required functions and classes from app.py
from talkinghead.tha3.app.app import (
    set_emotion,
    start_talking,
    stop_talking,
    talkinghead_load_file,
    launch,
    Animator,
    Encoder
)

# Global variables for animator and encoder
global_animator_instance = None
global_encoder_instance = None
animation_running = False
is_talking = False

# Default parameters
default_image_path = "/home/iby_vishwa/Documents/SillyTavern-extras/talkinghead/tha3/images/Jessie_image.png"
default_audio_path = "/home/iby_vishwa/Documents/SillyTavern-extras/talkinghead/final _malewav"
default_emotion = "anger"

def initialize_talking_head(device=0, model="standard_float"):
    """Initialize the talking head components."""
    global global_animator_instance
    global global_encoder_instance

    poser = load_poser(model, device, modelsdir="talkinghead/tha3/models")
    global_animator_instance = Animator(poser, device)
    global_encoder_instance = Encoder()

    global_animator_instance.load_image(default_image_path)
    global_animator_instance.start()
    global_encoder_instance.start()

def simulate_animation():
    """Simulate the animation by toggling start_talking and stop_talking."""
    global animation_running
    global is_talking

    set_emotion(default_emotion)
    talkinghead_load_file(open(default_image_path, 'rb'))

    # Simulate starting the talking animation
    start_talking()

    # Simulate animation duration based on audio length (example: 10 seconds)
    time.sleep(10)

    # Simulate stopping the talking animation
    stop_talking()

def capture_frames(duration=10, fps=25):
    """Capture frames for the specified duration and frame rate."""
    global animation_running
    frames = []

    def capture_frame():
        if global_animator_instance:
            frame = global_animator_instance.result_image
            if frame is not None:
                frames.append(frame)

    animation_running = True
    start_time = time.time()
    while time.time() - start_time < duration:
        capture_frame()
        time.sleep(1 / fps)
    
    animation_running = False
    return frames

def create_video(frames, output_path="output_video.mp4", fps=25):
    """Create a video from the captured frames."""
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()

def main():
    """Main function to run the modified server script."""
    initialize_talking_head(device=0, model="standard_float")

    # Simulate the animation process
    simulate_animation()

    # Capture frames during the animation
    frames = capture_frames(duration=10, fps=25)

    # Create a video from the captured frames
    create_video(frames, output_path="output_video.mp4", fps=25)

if __name__ == "__main__":
    main()
