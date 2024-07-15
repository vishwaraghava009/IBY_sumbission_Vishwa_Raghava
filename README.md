# SillyTavern - Extras
---
## Extras project is discontinued and won't receive any new updates or modules. The vast majority of modules are available natively in the main SillyTavern application. You may still install and use it but don't expect to get immediate support if you face any issues.
---
- [Recent news](#recent-news)
- [What is this](#what-is-this)
- [How to run](#how-to-run)
- [Modules](#modules)
- [Options](#options)
- [Coqui TTS](#coqui-tts)
- [ChromaDB](#chromadb)
- [API Endpoints](#api-endpoints)

## Recent news

* April 24 2024 - The project is officially discontinued.
* November 20 2023 - The project is relicensed as AGPLv3 to comply with the rest of ST organization policy. If you have any concerns about that, please raise a discussion in the appropriate channel.
* November 16 2023 - Requirement files were remade from scratch to simplify the process of local installation.
  - Removed requirements-complete.txt, please use requirements.txt instead.
  - Unlocked versions of all requirements unless strictly necessary.
  - Coqui TTS requirements moved to requirements-coqui.txt.
* July 25 2023 - Now extras require Python 3.11 to run, some of the new modules will be incompatible with old Python 3.10 installs. To migrate using conda, please remove old environment using `conda remove --name extras --all` and reinstall using the instructions below.

## What is this

A set of APIs for various SillyTavern extensions.

**You need to run the latest version of SillyTavern. Grab it here: [How to install](https://docs.sillytavern.app/installation/windows/), [Git repository](https://github.com/SillyTavern/SillyTavern)**

All modules, except for Stable Diffusion, run on the CPU by default. However, they can alternatively be configured to use CUDA (with `--cuda` command line option). When running all modules simultaneously, you can expect a usage of approximately 6 GB of RAM. Loading Stable Diffusion adds an additional couple of GB to the memory usage.

Some modules can be configured to use CUDA separately from the rest (e.g. `--talkinghead-gpu`, `--coqui-gpu` command line options). This is useful in low-VRAM setups, such as on a gaming laptop.

Try on Colab (will give you a link to Extras API):  <a target="_blank" href="https://colab.research.google.com/github/SillyTavern/SillyTavern/blob/release/colab/GPU.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Colab link:
https://colab.research.google.com/github/SillyTavern/SillyTavern/blob/release/colab/GPU.ipynb

Documentation:
https://docs.sillytavern.app/

## How to run
### :exclamation: **IMPORTANT!** Requirement files explained

* Default **requirements.txt** installs PyTorch CUDA by default.
* If you run on AMD GPU, use **requirements-rocm.txt** file instead.
* If you run on Apple Silicon (ARM series), use the **requirements-silicon.txt** file instead.
* If you want to use Coqui TTS, install **requirements-coqui.txt** after choosing the requirements from the list above.
* If you want to use RVC, install **requirements-rvc.txt** after choosing the requirements from the list above.
* BE WARNED THAT:
  - Coqui package is extremely unstable and may break other packages or not work at all in your environment.
  - It's not really worth it.

### Common errors when installing requirements

> *ERROR: Could not build wheels for hnswlib, which is required to install pyproject.toml-based projects*

Installing the chromadb package requires one of the following:

1. Have Visual C++ build tools installed: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Installing hnswlib from conda: `conda install -c conda-forge hnswlib`

**:exclamation: IMPORTANT!** The chromadb package is used **only** by the `chromadb` module for the old Smart Context extension, which is deprecated. You will likely not need it.

### Missing modules reported by SillyTavern extensions menu?

You must specify a list of module names to be run in the `--enable-modules` command (`caption` provided as an example). See [Modules](#modules) section.

### ☁️ Colab
* Open colab link
* Select desired "extra" options and start the cell
* Wait for it to finish
* Get an API URL link from colab output under the `### SillyTavern Extensions LINK ###` title
* Start SillyTavern with extensions support: set `enableExtensions` to `true` in config.conf
* Navigate to SillyTavern extensions menu and put in an API URL and tap "Connect" to load the extensions

### What about mobile/Android/Termux? 🤔

There are some folks in the community having success running Extras on their phones via Ubuntu on Termux. This project wasn't made with mobile support in mind, so this guide is provided strictly for your information only: https://rentry.org/STAI-Termux#downloading-and-running-tai-extras

#### :exclamation: IMPORTANT!

We will NOT provide any support for running Extras on Android. Direct all your questions to the creator of the guide linked above.

### 💻 Locally
#### Option 1 - Conda (recommended) 🐍

**PREREQUISITES**
* Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
* _(Important!) Read how to use Conda: https://conda.io/projects/conda/en/latest/user-guide/getting-started.html_
* Install git: https://git-scm.com/downloads

**EXECUTE THESE COMMANDS ONE BY ONE IN THE _CONDA COMMAND PROMPT_.**

**TYPE/PASTE EACH COMMAND INTO THE PROMPT, HIT ENTER AND WAIT FOR IT TO FINISH!**

* Before the first run, create an environment (let's call it `extras`):
```
conda create -n extras
```
* Now activate the newly created env
```
conda activate extras
```
* Install Python 3.11
```
conda install python=3.11
```
* Install the required system packages
```
conda install git
```
* Clone this repository
```
git clone https://github.com/SillyTavern/SillyTavern-extras
```
* Navigated to the freshly cloned repository
```
cd SillyTavern-extras
```
* Install the project requirements
```
pip install -r requirements.txt
```
* Run the Extensions API server
```
python server.py --enable-modules=caption,summarize,classify
```
* Copy the Extra's server API URL listed in the console window after it finishes loading up. On local installs, this defaults to `http://localhost:5100`.
* Open your SillyTavern config.conf file (located in the base install folder), and look for a line "`const enableExtensions`". Make sure that line has "`= true`", and not "`= false`".
* Start your SillyTavern server
* Open the Extensions panel (via the 'Stacked Blocks' icon at the top of the page), paste the API URL into the input box, and click "Connect" to connect to the Extras extension server.
* To run again, simply activate the environment and run these commands. Be sure to the additional options for server.py (see below) that your setup requires.
```
conda activate extras
python server.py
```

#### :exclamation: IMPORTANT! Talkinghead

Installation requirements for Talkinghead changed in January 2024. The live mode - i.e. the `talkinghead` module that powers the Talkinghead mode of Character Expressions - no longer needs any additional packages.

However, a manual poser app has been added, serving two purposes. First, it is a GUI editor for the Talkinghead emotion templates. Secondly, it can batch-generate static emotion sprites from a single Talkinghead image. The latter can be convenient if you want the convenience of AI-powered posing (e.g. if you make new characters often), but don't want to run the live mode.

The manual poser app, and **only** that app, still requires the installation of an additional package that is not installed automatically due to incompatibility with Colab. If you want to be able to use the manual poser app, then run this after you have installed other requirements:

```
conda activate extras
pip install wxpython==4.2.1
```

The installation of the wxpython package can easily take half an hour on a fast CPU, as it needs to compile a whole GUI toolkit.

More information about Talkinghead can be found in its [full documentation](talkinghead/README.md).

#### Option 2 - Vanilla 🍦
* Install Python 3.11: https://www.python.org/downloads/release/python-3114/
* Install git: https://git-scm.com/downloads
* Clone the repo:
```
git clone https://github.com/SillyTavern/SillyTavern-extras
cd SillyTavern-extras
```
* Run `python -m pip install -r requirements.txt`
* Run `python server.py --enable-modules=caption,summarize,classify`
* Get the API URL. Defaults to `http://localhost:5100` if you run locally.
* Start SillyTavern with extensions support: set `enableExtensions` to `true` in config.conf
* Navigate to the SillyTavern extensions menu and put in an API URL and tap "Connect" to load the extensions

## Modules

| Name          | Used by                                                              | Description                                                                      |
|---------------|----------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `caption`     |                                                                      | Image captioning                                                                 |
| `chromadb`    | [*Smart Context*](https://github.com/SillyTavern/Extension-ChromaDB) | Vector storage server                                                            |
| `classify`    | *Character Expressions*                                              | Text sentiment classification                                                    |
| `coqui-tts`   |                                                                      | [Coqui TTS server](https://github.com/coqui-ai/TTS)                              |
| `edge-tts`    |                                                                      | [Microsoft Edge TTS client](https://github.com/rany2/edge-tts)                   |
| `embeddings`  | *Vector Storage*                                                     | The *Extras* vectorization source                                                |
| `rvc`         |                                                                      | Real-time voice cloning                                                          |
| `sd`          |                                                                      | Stable Diffusion image generation (remote A1111 server by default)               |
| `silero-tts`  |                                                                      | [Silero TTS server](https://github.com/ouoertheo/silero-api-server)              |
| `summarize`   | *Summarize*                                                          | The *Extras API* backend                                                         |
| `talkinghead` | *Character Expressions*                                              | AI-powered character animation (see [full documentation](talkinghead/README.md)) |
| `websearch`   | [*Websearch*](https://github.com/SillyTavern/Extension-WebSearch)    | Google or DuckDuckGo search using Selenium headless browser                      |

#### **:exclamation: IMPORTANT!**

- *Character Expressions* can connect to two Extras modules, `classify` and `talkinghead`.
  - `classify` updates the expression of the AI character's avatar automatically based on text sentiment analysis.
  - `talkinghead` provides AI-powered character animation. It also takes its expression from the Extras `classify`.
    - To use Talkinghead, *Extensions ⊳ Character Expressions ⊳ Local server classification* in the ST GUI must be **off**, and `classify` must be enabled in Extras.
- *Smart Context* is deprecated; superseded by *Vector Storage*.
  - The `embeddings` module makes the ingestion performance comparable with ChromaDB, as it uses the same vectorization backend.
  - *Vector Storage* does not use other Extras modules.
- *Summarize*: the *Main API* is generally more capable, as it uses your main LLM to perform the summarization.
  - The `summarize` module is only used when you summarize with the *Extras API*. It uses a specialized BART summarization model, with a context size of 1024.

## Options

| Flag                     | Description                                                            |
| ------------------------ | ---------------------------------------------------------------------- |
| `--enable-modules`       | **Required option**. Which modules to enable.<br>Expects a comma-separated list of module names. Ordering does not matter. See [Modules](#modules)<br>Example: `--enable-modules=caption,sd` |
| `--port`                 | Specify the port on which the application is hosted. Default: **5100** |
| `--listen`               | Host the app on the local network                                      |
| `--share`                | Share the app on CloudFlare tunnel                                     |
| `--secure`               | Adds API key authentication requirements. Highly recommended when paired with share! |
| `--cpu`                  | Run the models on the CPU instead of CUDA. Enabled by default. |
| `--mps` or `--m1`        | Run the models on Apple Silicon. Only for M1 and M2 processors. |
| `--cuda`                 | Use CUDA (GPU+VRAM) to run modules if it is available. Otherwise, falls back to using CPU. |
| `--cuda-device`          | Specifies a CUDA device to use. Defaults to `cuda:0` (first available GPU). |
| `--talkinghead-gpu`      | Use CUDA (GPU+VRAM) for Talkinghead. **Highly recommended**, 10-30x FPS increase in animation. |
| `--talkinghead-model`    | Load a specific variant of the THA3 AI poser model for Talkinghead.<br>Default: `auto` (which is `separable_half` on GPU, `separable_float` on CPU). |
| `--talkinghead-models`   | If the THA3 AI poser models are not yet installed, downloads and installs them.<br>Expects a HuggingFace model ID.<br>Default: [OktayAlpk/talking-head-anime-3](https://huggingface.co/OktayAlpk/talking-head-anime-3) |
| `--coqui-gpu`            | Use GPU for coqui TTS (if available). |
| `--coqui-model`          | If provided, downloads and preloads a coqui TTS model. Default: none.<br>Example: `tts_models/multilingual/multi-dataset/bark` |
| `--summarization-model`  | Load a custom summarization model.<br>Expects a HuggingFace model ID.<br>Default: [Qiliang/bart-large-cnn-samsum-ChatGPT_v3](https://huggingface.co/Qiliang/bart-large-cnn-samsum-ChatGPT_v3) |
| `--classification-model` | Load a custom sentiment classification model.<br>Expects a HuggingFace model ID.<br>Default (6 emotions): [nateraw/bert-base-uncased-emotion](https://huggingface.co/nateraw/bert-base-uncased-emotion)<br>Other solid option is (28 emotions): [joeddav/distilbert-base-uncased-go-emotions-student](https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions-student)<br>For Chinese language: [touch20032003/xuyuan-trial-sentiment-bert-chinese](https://huggingface.co/touch20032003/xuyuan-trial-sentiment-bert-chinese) |
| `--captioning-model`     | Load a custom captioning model.<br>Expects a HuggingFace model ID.<br>Default: [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) |
| `--embedding-model`      | Load a custom text embedding (vectorization) model. Both the `embeddings` and `chromadb` modules use this.<br>The backend is [`sentence_transformers`](https://pypi.org/project/sentence-transformers/), so check there for info on supported models.<br>Expects a HuggingFace model ID.<br>Default: [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) |
| `--chroma-host`          | Specifies a host IP for a remote ChromaDB server. |
| `--chroma-port`          | Specifies an HTTP port for a remote ChromaDB server.<br>Default: `8000` |
| `--sd-model`             | Load a custom Stable Diffusion image generation model.<br>Expects a HuggingFace model ID.<br>Default: [ckpt/anything-v4.5-vae-swapped](https://huggingface.co/ckpt/anything-v4.5-vae-swapped)<br>*Must have VAE pre-baked in PyTorch format or the output will look drab!* |
| `--sd-cpu`               | Force the Stable Diffusion generation pipeline to run on the CPU.<br>**SLOW!** |
| `--sd-remote`            | Use a remote SD backend.<br>**Supported APIs: [sd-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)**  |
| `--sd-remote-host`       | Specify the host of the remote SD backend<br>Default: **127.0.0.1** |
| `--sd-remote-port`       | Specify the port of the remote SD backend<br>Default: **7860** |
| `--sd-remote-ssl`        | Use SSL for the remote SD backend<br>Default: **False** |
| `--sd-remote-auth`       | Specify the `username:password` for the remote SD backend (if required) |

## Coqui TTS

### Running on Mac M1

#### ImportError: symbol not found

If you're getting the following error when running coqui-tts module on M1 Mac:

```
ImportError: dlopen(/Users/user/.../lib/python3.11/site-packages/MeCab/_MeCab.cpython-311-darwin.so, 0x0002): symbol not found in flat namespace '__ZN5MeCab11createModelEPKc'
```

Do the following:

1. Install homebrew: https://brew.sh/
2. Build and install the `mecab` package

```
brew install --build-from-source mecab
ARCHFLAGS='-arch arm64' pip install --no-binary :all: --compile --use-pep517 --no-cache-dir --force mecab-python3
```

## ChromaDB

**:exclamation: IMPORTANT!** ChromaDB is used **only** by the `chromadb` module for the old Smart Context extension, which is deprecated. You will likely not need it.

ChromaDB is a blazing fast and open source database that is used for long-term memory when chatting with characters. It can be run in-memory or on a local server on your LAN.

NOTE: You should **NOT** run ChromaDB on a cloud server. There are no methods for authentication (yet), so unless you want to expose an unauthenticated ChromaDB to the world, run this on a local server in your LAN.

### In-memory setup

Run the extras server with the `chromadb` module enabled (recommended).

### Remote setup

Use this if you want to use ChromaDB with docker or host it remotely. If you don't know what that means and only want to use ChromaDB with ST on your local device, use the 'in-memory' instructions instead.

Prerequisites: Docker, Docker compose (make sure you're running in rootless mode with the systemd service enabled if on Linux).

Steps:

1. Run `git clone https://github.com/chroma-core/chroma chromadb` and `cd chromadb`
2. Run `docker-compose up -d --build` to build ChromaDB. This may take a long time depending on your system
3. Once the build process is finished, ChromaDB should be running in the background. You can check with the command `docker ps`
4. On your client machine, specify your local server ip in the `--chroma-host` argument (ex. `--chroma-host=192.168.1.10`)


If you are running ChromaDB on the same machine as SillyTavern, you will have to change the port of one of the services. To do this for ChromaDB:

1. Run `docker ps` to get the container ID and then `docker container stop <container ID>`
2. Enter the ChromaDB git repository `cd chromadb`
3. Open `docker-compose.yml` and look for the line starting with `uvicorn chromadb.app:app`
4. Change the `--port` argument to whatever port you want.
5. Look for the `ports` category and change the occurrences of `8000` to whatever port you chose in step 4.
6. Save and exit. Then run `docker-compose up --detach`
7. On your client machine, make sure to specity the `--chroma-port` argument (ex. `--chroma-port=<your-port-here>`) along with the `--chroma-host` argument.

## API Endpoints

*This section is developer documentation, containing usage examples of the API endpoints.*

*This is kept up-to-date on a best-effort basis, but there is a risk of this documentation being out of date. When in doubt, refer to the actual source code.*

### Get list of enabled modules
`GET /api/modules`
#### **Input**
None
#### **Output**
```
{"modules":["caption", "classify", "summarize"]}
```

### Image captioning
`POST /api/caption`
#### **Input**
```
{ "image": "base64 encoded image" }
```
#### **Output**
```
{ "caption": "caption of the posted image" }
```

### Text summarization
`POST /api/summarize`
#### **Input**
```
{ "text": "text to be summarize", "params": {} }
```
#### **Output**
```
{ "summary": "summarized text" }
```
#### Optional: `params` object for control over summarization:
| Name                  | Default value                                                 |
| --------------------- | ------------------------------------------------------------- |
| `temperature`         | 1.0                                                           |
| `repetition_penalty`  | 1.0                                                           |
| `max_length`          | 500                                                           |
| `min_length`          | 200                                                           |
| `length_penalty`      | 1.5                                                           |
| `bad_words`           | ["\n", '"', "*", "[", "]", "{", "}", ":", "(", ")", "<", ">"] |

### Text sentiment classification
`POST /api/classify`
#### **Input**
```
{ "text": "text to classify sentiment of" }
```
#### **Output**
```
{
    "classification": [
        {
            "label": "joy",
            "score": 1.0
        },
        {
            "label": "anger",
            "score": 0.7
        },
        {
            "label": "love",
            "score": 0.6
        },
        {
            "label": "sadness",
            "score": 0.5
        },
        {
            "label": "fear",
            "score": 0.4
        },
        {
            "label": "surprise",
            "score": 0.3
        }
    ]
}
```
> **NOTES**
> 1. Sorted by descending score order
> 2. List of categories defined by the summarization model
> 3. Value range from 0.0 to 1.0

### Stable Diffusion image generation
`POST /api/image`
#### **Input**
```
{ "prompt": "prompt to be generated", "sampler": "DDIM", "steps": 20, "scale": 6, "model": "model_name" }
```
#### **Output**
```
{ "image": "base64 encoded image" }
```
> **NOTES**
> 1. Only the "prompt" parameter is required
> 2. Both "sampler" and "model" parameters only work when using a remote SD backend

### Get available Stable Diffusion models
`GET /api/image/models`
#### **Output**
```
{ "models": [list of all available model names] }
```

### Get available Stable Diffusion samplers
`GET /api/image/samplers`
#### **Output**
```
{ "samplers": [list of all available sampler names] }
```

### Get currently loaded Stable Diffusion model
`GET /api/image/model`
#### **Output**
```
{ "model": "name of the current loaded model" }
```

### Load a Stable Diffusion model (remote)
`POST /api/image/model`
#### **Input**
```
{ "model": "name of the model to load" }
```
#### **Output**
```
{ "previous_model": "name of the previous model", "current_model": "name of the newly loaded model" }
```

### Generate Silero TTS voice
`POST /api/tts/generate`
#### **Input**
```
{ "speaker": "speaker voice_id", "text": "text to narrate" }
```
#### **Output**
WAV audio file.

### Get Silero TTS voices
`GET /api/tts/speakers`
#### **Output**
```
[
    {
        "name": "en_0",
        "preview_url": "http://127.0.0.1:5100/api/tts/sample/en_0",
        "voice_id": "en_0"
    }
]
```

### Get Silero TTS voice sample
`GET /api/tts/sample/<voice_id>`
#### **Output**
WAV audio file.


### Compute text embeddings (vectorize)
`POST /api/embeddings/compute`

This is a vectorization source (text embedding provider) for the Vector Storage built-in extension of ST.

If you have many text items to vectorize (e.g. chat history, or chunks for RAG ingestion), send them in all at once. This allows the backend to batch the input, allocating the available compute resources efficiently, and thus running much faster (compared to processing a single item at a time).

The embeddings are always normalized.

#### **Input**
For one text item:
```
{ "text": "The quick brown fox jumps over the lazy dog." }
```
For multiple text items, just put them in an array:
```
{ "text": ["The quick brown fox jumps over the lazy dog.",
           "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
           ...] }
```
#### **Output**
When the input was one text item, returns one vector (the embedding of that text item) as an array:
```
{ "embedding": [numbers] }
```
When the input was multiple text items, returns multiple vectors (one for each input text item) as an array of arrays:
```
{ "embedding": [[numbers],
                [numbers], ...] }
```


### Add messages to chromadb
`POST /api/chromadb`
#### **Input**
```
{
    "chat_id": "chat1 - 2023-12-31",
    "messages": [
        {
            "id": "633a4bd1-8350-46b5-9ef2-f5d27acdecb7",
            "date": 1684164339877,
            "role": "user",
            "content": "Hello, AI world!",
            "meta": "this is meta"
        },
        {
            "id": "8a2ed36b-c212-4a1b-84a3-0ffbe0896506",
            "date": 1684164411759,
            "role": "assistant",
            "content": "Hello, Hooman!"
        },
    ]
}
```
#### **Output**
```
{ "count": 2 }
```

### Query chromadb
`POST /api/chromadb/query`
#### **Input**
```
{
    "chat_id": "chat1 - 2023-12-31",
    "query": "Hello",
    "n_results": 2,
}
```
#### **Output**
```
[
    {
        "id": "633a4bd1-8350-46b5-9ef2-f5d27acdecb7",
        "date": 1684164339877,
        "role": "user",
        "content": "Hello, AI world!",
        "distance": 0.31,
        "meta": "this is meta"
    },
    {
        "id": "8a2ed36b-c212-4a1b-84a3-0ffbe0896506",
        "date": 1684164411759,
        "role": "assistant",
        "content": "Hello, Hooman!",
        "distance": 0.29
    },
]
```

### Delete the messages from chromadb
`POST /api/chromadb/purge`
#### **Input**
```
{ "chat_id": "chat1 - 2023-04-12" }
```

### Get a list of Edge TTS voices
`GET /api/edge-tts/list`
#### **Output**
```
[{'Name': 'Microsoft Server Speech Text to Speech Voice (af-ZA, AdriNeural)', 'ShortName': 'af-ZA-AdriNeural', 'Gender': 'Female', 'Locale': 'af-ZA', 'SuggestedCodec': 'audio-24khz-48kbitrate-mono-mp3', 'FriendlyName': 'Microsoft Adri Online (Natural) - Afrikaans (South Africa)', 'Status': 'GA', 'VoiceTag': {'ContentCategories': ['General'], 'VoicePersonalities': ['Friendly', 'Positive']}}]
```

### Generate Edge TTS voice
`POST /api/edge-tts/generate`
#### **Input**
```
{ "text": "Text to narrate", "voice": "af-ZA-AdriNeural", "rate": 0 }
```
#### **Output**
MP3 audio file.

### Load a Coqui TTS model
`GET /api/coqui-tts/load`
#### **Input**
_model (string, required): The name of the Coqui TTS model to load.
_gpu (string, Optional): Use the GPU to load model.
_progress (string, Optional): Show progress bar in terminal.
```
{ "_model": "tts_models--en--jenny--jenny\model.pth" }
{ "_gpu": "False" }
{ "_progress": "True" }
```
#### **Output**
"Loaded"

### Get a list of Coqui TTS voices
`GET /api/coqui-tts/list`
#### **Output**
```
["tts_models--en--jenny--jenny\\model.pth", "tts_models--en--ljspeech--fast_pitch\\model_file.pth", "tts_models--en--ljspeech--glow-tts\\model_file.pth", "tts_models--en--ljspeech--neural_hmm\\model_file.pth", "tts_models--en--ljspeech--speedy-speech\\model_file.pth", "tts_models--en--ljspeech--tacotron2-DDC\\model_file.pth", "tts_models--en--ljspeech--vits\\model_file.pth", "tts_models--en--ljspeech--vits--neon\\model_file.pth.tar", "tts_models--en--multi-dataset--tortoise-v2", "tts_models--en--vctk--vits\\model_file.pth", "tts_models--et--cv--vits\\model_file.pth.tar", "tts_models--multilingual--multi-dataset--bark", "tts_models--multilingual--multi-dataset--your_tts\\model_file.pth", "tts_models--multilingual--multi-dataset--your_tts\\model_se.pth"]
```

### Get a list of the loaded Coqui model speakers
`GET /api/coqui-tts/multspeaker`
#### **Output**
```
{"0": "female-en-5", "1": "female-en-5\n", "2": "female-pt-4\n", "3": "male-en-2", "4": "male-en-2\n", "5": "male-pt-3\n"}
```

### Get a list of the loaded Coqui model lanagauges
`GET /api/coqui-tts/multlang`
#### **Output**
```
{"0": "en", "1": "fr-fr", "2": "pt-br"}
```

### Generate Coqui TTS voice
`POST /api/edge-tts/generate`
#### **Input**
```
{
  "text": "Text to narrate",
  "speaker_id": "0",
  "mspker": null,
  "language_id": null,
  "style_wav": null
}
```
#### **Output**
MP3 audio file.

### Load a talkinghead character
`POST /api/talkinghead/load`
#### **Input**
A `FormData` with files, with an image file in a field named `"file"`. The posted file should be a PNG image in RGBA format. Optimal resolution is 512x512. See the [`talkinghead` README](talkinghead/README.md) for details.
#### **Example**
'http://localhost:5100/api/talkinghead/load'
#### **Output**
'OK'

### Load talkinghead emotion templates (or reset them to defaults)
`POST /api/talkinghead/load_emotion_templates`
#### **Input**
```
{"anger": {"eyebrow_angry_left_index": 1.0,
           ...}
 "curiosity": {"eyebrow_lowered_left_index": 0.5895,
               ...}
 ...}
```
For details, see `Animator.load_emotion_templates` in [`talkinghead/tha3/app/app.py`](talkinghead/tha3/app/app.py). This is essentially the format used by [`talkinghead/emotions/_defaults.json`](talkinghead/emotions/_defaults.json).

Any emotions NOT supplied in the posted JSON will revert to server defaults. In any supplied emotion, any morph NOT supplied will default to zero. This allows making the templates shorter.

To reset all emotion templates to their server defaults, send a blank JSON.
#### **Output**
"OK"

### Load talkinghead animator/postprocessor settings (or reset them to defaults)
`POST /api/talkinghead/load_animator_settings`
#### **Input**
```
{"target_fps": 25,
 "breathing_cycle_duration": 4.0,
 "postprocessor_chain": [["bloom", {}],
                         ["chromatic_aberration", {}],
                         ["vignetting", {}],
                         ["translucency", {"alpha": 0.9}],
                         ["alphanoise", {"magnitude": 0.1, "sigma": 0.0}],
                         ["banding", {}],
                         ["scanlines", {"dynamic": true}]]
 ...}
```
For a full list of supported settings, see `animator_defaults` and `Animator.load_animator_settings`, both in [`talkinghead/tha3/app/app.py`](talkinghead/tha3/app/app.py).

Particularly for `"postprocess_chain"`, see [`talkinghead/tha3/app/postprocessor.py`](talkinghead/tha3/app/postprocessor.py). The postprocessor applies pixel-space glitch artistry, which can e.g. make your talkinghead look like a scifi hologram (the above example does this). The postprocessing filters are applied in the order they appear in the list.

To reset all animator/postprocessor settings to their server defaults, send a blank JSON.
#### **Output**
"OK"

### Animate the talkinghead character to start talking
`GET /api/talkinghead/start_talking`
#### **Example**
'http://localhost:5100/api/talkinghead/start_talking'
#### **Output**
"talking started"

### Animate the talkinghead character to stop talking
`GET /api/talkinghead/stop_talking`
#### **Example**
'http://localhost:5100/api/talkinghead/stop_talking'
#### **Output**
"talking stopped"

### Set the talkinghead character's emotion
`POST /api/talkinghead/set_emotion`

Available emotions: see `talkinghead/emotions/*.json`. An emotion must be specified, but if it is not available, this operation defaults to `"neutral"`, which must always be available. This endpoint is the backend behind the `/emote` slash command in talkinghead mode.
#### **Input**
```
{"emotion_name": "curiosity"}
```
#### **Example**
'http://localhost:5100/api/talkinghead/set_emotion'
#### **Output**
"emotion set to curiosity"

### Output the animated talkinghead sprite.
`GET /api/talkinghead/result_feed`
#### **Output**
Animated transparent image, each frame a 512x512 PNG image in RGBA format.

### Perform web search
`POST /api/websearch`

Available engines: `google` (default), `duckduckgo`
#### **Input**
```
{ "query": "what is beauty?", "engine": "google" }
```
#### **Output**
```
{ "results": "that would fall within the purview of your conundrums of philosophy", "links": ["http://example.com"] }
```
