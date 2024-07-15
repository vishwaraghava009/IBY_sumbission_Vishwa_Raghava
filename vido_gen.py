import requests
import json

# Define the server URL
server_url = "http://localhost:5100/api/generate_video"

# Prepare the request payload
payload = {
    "audio_file_path": "/home/iby_vishwa/Documents/SillyTavern-extras/modules/text_to_speech/alltalk_tts/outputs/myoutputfile.wav",
    "animator_settings": {
        "fps": 30,
        "target_fps": 25,
        "crop_left": 0.0,
        "crop_right": 0.0,
        "crop_top": 0.0,
        "crop_bottom": 0.0,
        "pose_interpolator_step": 0.1,
        "blink_interval_min": 2.0,
        "blink_interval_max": 5.0,
        "blink_probability": 0.03,
        "blink_confusion_duration": 10.0,
        "talking_fps": 12,
        "talking_morph": "mouth_aaa_index",
        "sway_morphs": ["head_x_index", "head_y_index", "neck_z_index", "body_y_index", "body_z_index"],
        "sway_interval_min": 5.0,
        "sway_interval_max": 10.0,
        "sway_macro_strength": 0.6,
        "sway_micro_strength": 0.02,
        "breathing_cycle_duration": 4.0,
        "postprocessor_chain": []
    }
}

# Make the POST request
response = requests.post(server_url, json=payload)

# Handle the response
if response.status_code == 200:
    with open('output_video.mp4', 'wb') as f:
        f.write(response.content)
    print("Video generated and saved as output_video.mp4")
else:
    print(f"Failed to generate video: {response.status_code}")
    print(response.text)
