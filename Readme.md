# Simple Pipeline for Audio Driven Human Talking Heads

This repository provides a simple pipeline for generating audio-driven talking heads using two models:

1. **AdaSR Talking Head**: For motion transfer or pose transfer, which includes transferring expressions, eye-blinks, and head movements.
2. **MuseTalk**: For lip synchronization.
   
## Prerequisites for Installation
- Install Anaconda or Miniconda.
- Install git.
  
## Steps for Installation
1. **Clone this branch and change the directory to `Simple_Pipeline` or download the zip file from this branch:**
   ```sh
   git clone -b Simple_Pipeline https://github.com/vishwaraghava009/IBY_sumbission_Vishwa_Raghava.git
   cd Simple_Pipeline
   
2. **Create and activate an environment for AdaSR:**
   ```sh
   cd AdaSR-TalkingHead
   conda env create -f environment.yaml
   cd ..
   
3. **Create and activate an environment for LivePortrait:**
   ```sh
   cd LivePortrait
   conda create -n liveportrait
   pip install -r requirements.txt
   cd ..

4. **Create and activate an environment for MuseTalk:**
   ```sh
   cd MuseTalk
   conda create -n musetalk
   conda install python=3.10
   pip install -r requirements.txt
   pip install --no-cache-dir -U openmim 
   mim install mmengine 
   mim install "mmcv>=2.0.1" 
   mim install "mmdet>=3.1.0" 
   mim install "mmpose>=1.1.0"
   export FFMPEG_PATH=/path/to/ffmpeg  #for example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static
   cd ..

## Checkpoints

1. **Checkpoints for AdaSR-Talking Head**: [https://drive.google.com/file/d/1g58uuAyZFdny9_twvbv0AHxB9-03koko/view?usp=sharing](https://drive.google.com/file/d/1g58uuAyZFdny9_twvbv0AHxB9-03koko/view?usp=sharing)
   - Download, extract and put it under checkpoints/

2. **Checkpoints for LivePortrait**
   ```sh
   # you may need to run `git lfs install` first
   git clone https://huggingface.co/KwaiVGI/liveportrait pretrained_weights
   
3. **Checkpoints for MuseTalk**: [https://drive.google.com/drive/folders/1r629pRzt54hft7GGPZdAwKKFJF3jmvQP?usp=sharing](https://drive.google.com/drive/folders/1r629pRzt54hft7GGPZdAwKKFJF3jmvQP?usp=sharing)
   - You have to arrange these models in the same structure under models/
```
./models/
├── musetalk
│   └── musetalk.json
│   └── pytorch_model.bin
├── dwpose
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── sd-vae-ft-mse
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── whisper
    └── tiny.pt
```

## Steps to Run the Pipeline

### Instructions

1. **Login to the profile: `iby_vishwa`.**

2. **Change the directory to `Simple_Pipeline`**
   
   ```sh
   cd Simple_Pipeline

4. **Upload an image and an audio file (.wav) of your choice** into the `DEMO` subdirectory of the `AdaSR-TalkingHead` directory.

5. **Edit the `run_demo.sh` file:**
   - Open `run_demo.sh` in the `AdaSR-TalkingHead` directory.
   - Edit the image, audio and driving video file names according to your uploaded files.

6. **Run the following commands to perform motion transfer:**

    ```sh
    conda activate mesh-video
    cd AdaSR-TalkingHead
    bash run_demo.sh
    ```

    - The motion transfer video will be saved in the same directory.

7. **Deactivate the conda environments:**

    ```sh
    conda deactivate
    cd ..
    ```

8. **Run the first inference of MuseTalk:**

    ```sh
    conda activate musetalk_env
    cd MuseTalk
    bash infer.sh
    ```

    - The results will be saved under the directory `Musetalk/results/avatars/avator_1/vid_output`.
    - Make sure you delete the existing avatar to test on another character.
  
9. **LivePortrait an alternative Motion Transfer model:**

   ```sh
   cd LivePortrait
   conda activate liveportrait
   python inference.py -s /path/to/your/image -d /path/to/your/driving_video --no_flag_pasteback

### Note

- The first inference might take some extra time to create avatars (facial volumetric features). These can be saved beforehand to reduce inference time for subsequent runs.
- Mke sure you cahnge the paths of directories in the `run_demo.sh` and `infer.sh` files.

## Directory Structure

.                                                                                                                                              
├── AdaSR-TalkingHead                                                                                                                          
│   ├── DEMO                                                                                                                                   
│   │   ├── <your_image>.jpeg                                                                                                                  
│   │   ├── <your_audio>.wav                                                                                                                   
│   ├── run_demo.sh                                                                                                                            
│   ├── ...                                                                                                                                    
│                                                                                                                                              
├── MuseTalk                                                                                                                                   
│   ├── configs                                                                                                                                
│   │   ├── inference                                                                                                                          
│   │   │   ├── realtime.yaml                                                                                                                  
│   ├── scripts                                                                                                                                
│   │   ├── realtime_inference.py                                                                                                              
│   ├── infer.sh                                                                                                                               
│   ├── ...               

Make sure to replace `<your_image>.jpeg` and `<your_audio>.wav` with your actual files in the `DEMO` directory and `run_demo.sh`.

## Troubleshooting

If you encounter any issues, please look into the following repositories:
   - AdaSR : [https://github.com/Songluchuan/AdaSR-TalkingHead.git](https://github.com/Songluchuan/AdaSR-TalkingHead.git)
   - LivePortrait : [https://github.com/KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait)
   - MuseTalk : [https://github.com/TMElyralab/MuseTalk.git](https://github.com/TMElyralab/MuseTalk.git)
