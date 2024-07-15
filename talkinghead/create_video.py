import io
import os
import time
import torch
import random
import threading
import numpy as np
import wave
import contextlib
from PIL import Image
from typing import Any, Dict, List, Optional, Union
from tha3.app.app import (Animator, Encoder, logger, load_poser, posedict_key_to_index, posedict_to_pose, _animator_output_lock)
from tha3.poser.modes.load_poser import load_poser
from tha3.util import (torch_linear_to_srgb, resize_PIL_image, extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image)
import subprocess

class VideoGenerator:
    def __init__(self, device: str, audio_path: str, output_dir: str, output_video: str, fps: int = 25):
        self.device = device
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.output_video = output_video
        self.fps = fps
        self.animator = None
        self.encoder = None
        self.is_talking = False

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_audio_duration(self) -> float:
        with contextlib.closing(wave.open(self.audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        return duration

    def initialize_animator(self):
        logger.info("Initializing animator")
        model_dir = "tha3/models"
        model_name = os.listdir(model_dir)[0]  # Assuming there's only one model in the directory
        poser = load_poser(model_name, self.device, modelsdir=model_dir)
        self.animator = Animator(poser, self.device)
        self.encoder = Encoder()
        initial_image_path = os.path.join(os.getcwd(), "tha3/images/initial.png")
        self.animator.load_image(initial_image_path)
        self.animator.start()
        self.encoder.start()


    def save_frame(self, frame_count: int):
        with _animator_output_lock:
            image_rgba = self.animator.result_image
            self.animator.new_frame_available = False

        pil_image = Image.fromarray(np.uint8(image_rgba[:, :, :3]))
        if image_rgba.shape[2] == 4:
            alpha_channel = image_rgba[:, :, 3]
            pil_image.putalpha(Image.fromarray(np.uint8(alpha_channel)))

        frame_path = os.path.join(self.output_dir, f"frame_{frame_count:05d}.png")
        pil_image.save(frame_path)

    def generate_frames(self, duration: float):
        frame_count = 0
        total_frames = int(self.fps * duration)

        for frame_count in range(total_frames):
            self.animator.render_animation_frame()
            if self.animator.new_frame_available:
                self.save_frame(frame_count)
            time.sleep(1 / self.fps)

    def compile_video(self):
        ffmpeg_cmd = [
            'ffmpeg', '-framerate', str(self.fps), '-i', os.path.join(self.output_dir, 'frame_%05d.png'),
            '-i', self.audio_path, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', self.output_video
        ]
        subprocess.run(ffmpeg_cmd)

    def create_video_from_audio(self):
        duration = self.load_audio_duration()
        self.initialize_animator()
        self.generate_frames(duration)
        self.compile_video()

if __name__ == "__main__":
    device = "cuda"  # or "cpu"
    audio_path = "/home/iby_vishwa/Documents/SillyTavern-extras/talkinghead/final _male.wav"
    output_dir = "output_frames"
    output_video = "output_video.mp4"

    video_generator = VideoGenerator(device, audio_path, output_dir, output_video)
    video_generator.create_video_from_audio()
