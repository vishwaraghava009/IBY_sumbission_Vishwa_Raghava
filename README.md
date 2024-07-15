
# Generating Animating Talking Heads in One Shot

This branch is the modified work from SillyTavern-Extras. It has been enhanced to include AllTalk TTS, align the animator with the generated audio file, make API calls to the llama3-8b-8192 model from Groq, and include a simple UI for inference purposes.

## Prerequisites for Installation
1. Install Anaconda or Miniconda.
2. Install git.

## Installation

1. **Create and activate an environment:**
   ```sh
   conda create -n anime
   conda activate anime
   ```

2. **Install Python 3.11 and git:**
   ```sh
   conda install python=3.11
   conda install git
   ```

3. **Clone this branch and change the directory to `Animated_talkingHead` or download the zip file from this branch:**
   ```sh
   git clone -b Animated_TalkingHead https://github.com/vishwaraghava009/IBY_sumbission_Vishwa_Raghava.git
   cd Animated_talkingHead
   ```

4. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

5. **Install AllTalk TTS:**
   ```sh
   cd modules/text_to_speech/alltalk_tts
   bash atsetup.sh
   ```

6. **You'll see a terminal with 2 options; select `option 2` as we are installing it as a standalone application.**

7. **Next, you'll get 9 more options, choose `option 1`. Wait until the terminal says `press any key to exit`.**

## Starting the Servers for Talking Head and AllTalk TTS Modules

1. **To start the server for AllTalk TTS, change the directory to `modules/text_to_speech/alltalk_tts` and run the following command:**
   ```sh
   bash start_alltalk.sh
   ```
   This triggers `script.py` and gets hosted at port 7851.

2. **To start the Talking Head server, change the directory to `Animated_TalkingHead` and run the following command:**
   ```sh
   python server.py --enable-modules=classify,talkinghead --classification-model=joeddav/distilbert-base-uncased-go-emotions-student --talkinghead-gpu
   ```
   This makes it get hosted at port 5100.

## Inference

### To Infer the Live Streaming of the Talking Head

**Prerequisites:**
1. Make sure you started both the above-mentioned servers.
2. Change the Groq API key in `server.py` line`380` with your own Groq API key.

1. **Change the directory to `modules/text_to_speech/alltalk_tts/outputs` and run the following command:**
   ```sh
   python serve_with_cors.py
   ```
   This hosts our UI at port 8000.

2. **Navigate to port 8000 through your browser and upload the `talkinghead.png` from the directory using the upload button.**
   - If you want to try with your custom image, you have to edit your image to match the size ratios of the provided image and should have no background. Importantly, the head size should align exactly with the provided image.

3. **Choose your language from the dropdown and start typing the message.**

## References

1. **If you find difficulties with any of the installations mentioned above, please refer to the following repositories:**
   - SillyTavern-Extras: [https://github.com/SillyTavern/SillyTavern-Extras.git](https://github.com/SillyTavern/SillyTavern-Extras.git)
   - AllTalk TTS: [https://github.com/erew123/alltalk_tts.git](https://github.com/erew123/alltalk_tts.git)

2. **If you find any difficulty in inference and API endpoints, please refer to the documents I submitted.**

