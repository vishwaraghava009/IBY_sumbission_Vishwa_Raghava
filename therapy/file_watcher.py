# file_watcher.py
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import yaml

class Watcher:
    DIRECTORY_TO_WATCH = "Virtual-Therapist/media/audio"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()

class Handler(FileSystemEventHandler):

    @staticmethod
    def process(event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Run the motion transfer and MuseTalk inference
            audio_file_path = event.src_path
            handle_new_audio_file(audio_file_path)

    def on_created(self, event):
        self.process(event)

def handle_new_audio_file(audio_file_path):
    # Path to the motion transferred video directory
    motion_video_dir = "Virtual-Therapist/media/motion_video/vox/"
    output_video_dir = "Virtual-Therapist/media/output_video/avatars/avator1/"
    
    # Wait until the motion transferred video is available
    while True:
        motion_videos = [f for f in os.listdir(motion_video_dir) if f.endswith('.mp4')]
        if motion_videos:
            motion_video_path = os.path.join(motion_video_dir, motion_videos[0])
            break
        time.sleep(5)
    
    # Check if the avatar directory exists
    avatar_dir = "Virtual-Therapist/media/output_video/avatars/avator1/"
    preparation_needed = not os.path.exists(avatar_dir)
    
    # Update the realtime.yaml configuration file
    update_realtime_yaml(motion_video_path, audio_file_path, preparation_needed)
    
    # Run the MuseTalk inference script
    run_musetalk_inference()

def update_realtime_yaml(video_path, audio_path, preparation):
    config_path = "Virtual-Therapist/MuseTalk/configs/inference/realtime.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config['avator_1']['preparation'] = preparation
    config['avator_1']['video_path'] = video_path
    config['avator_1']['audio_clips']['audio_0'] = audio_path

    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)

def run_musetalk_inference():
    command = [
        'python', '-m', 'scripts.realtime_inference',
        '--inference_config', 'configs/inference/realtime.yaml'
    ]
    subprocess.run(command, cwd='Virtual-Therapist/MuseTalk')

if __name__ == '__main__':
    w = Watcher()
    w.run()
