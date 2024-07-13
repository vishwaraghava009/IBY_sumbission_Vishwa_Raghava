
# UI Integrated Human Talking Heads

This repository is meant for the UI integration of the Human Talking Head pipeline.

### Important Note:
- This pipeline does not include motion transfer models as we changed the plan to storing motion transferred video in our backend.
- Go through the python notebook or install the same libraries on your local machine as well by cloning the repo and by following the mentioned changes in the colab notebook.
- You may have to create a new Groq API key if mine doesn't work and replace mine in the `therapy/api_vies.py` file.
- You may also have to create an Ngrok auth number as well if mine doesn't work; this is specific only if you are using colab for execution.
- Recommended to use V100 GPU or advanced.

### Model Checkpoints:
- **MuseTalk:** [MuseTalk Model Checkpoints](https://drive.google.com/drive/folders/1r629pRzt54hft7GGPZdAwKKFJF3jmvQP?usp=sharing)
  - Please put the models directory in your drive in the same structure as the drive folders provided above if you are using colab.
  - If you are using your local Linux machine, try putting the MuseTalk models in the models directory in the same structure as provided in the drive link.
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
## Prerequisites for Installation
- Install Anaconda
- Install git

## Steps for Installation

1. **Clone this branch and change the directory to `Human_TalkingHead` or download the zip file from this branch:**
   ```sh
   git clone -b Human_TalkingHead https://github.com/vishwaraghava009/IBY_sumbission_Vishwa_Raghava.git
   cd Human_TalkingHead
   ```

2. **Create and activate an environment:**
   ```sh
   conda create -n env
   conda install python=3.10
   conda activate env
   apt-get install -y portaudio19-dev
   apt-get install ffmpeg
   ```

3. **Clone StyleTTS2 and install its dependencies:**
   ```sh
   git clone https://github.com/yl4579/StyleTTS2.git
   cd StyleTTS2
   pip install SoundFile torchaudio munch torch pydub pyyaml librosa nltk matplotlib accelerate transformers phonemizer einops einops-exts tqdm typing-extensions git+https://github.com/resemble-ai/monotonic_align.git
   sudo apt-get install espeak-ng
   git-lfs clone https://huggingface.co/yl4579/StyleTTS2-LJSpeech
   mv StyleTTS2-LJSpeech/Models .
   cd ..
   ```

4. **Install required dependencies to run the Django server:**
   ```sh
   pip install -r requirements.txt
   ```
   - You may have to install `dlib` if your machine doesn't support some of these dependencies.

5. **Clone MuseTalk and install its dependencies:**
   ```sh
   git clone https://github.com/TMElyralab/MuseTalk.git
   cd MuseTalk
   pip install -r requirements.txt
   pip install --no-cache-dir -U openmim 
   mim install mmengine 
   mim install "mmcv>=2.0.1" 
   mim install "mmdet>=3.1.0" 
   mim install "mmpose>=1.1.0"
   export FFMPEG_PATH=/path/to/ffmpeg  # for example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static
   cd ..
   ```

6. **Create the following script under `StyleTTS2/`:**
   ```python
   import sys
   import os
   import torch
   import random
   import numpy as np
   import yaml
   import torchaudio
   import librosa
   from munch import Munch
   from nltk.tokenize import word_tokenize
   import phonemizer
   import time
   import nltk
   nltk.download('punkt')

   # Ensure current directory is the script's directory
   os.chdir(os.path.dirname(os.path.abspath(__file__)))

   # Load the LLM output and audio filename from the command line arguments
   llm_output_path = os.path.abspath(sys.argv[1])
   audio_filename = os.path.abspath(sys.argv[2])

   # Initialize random seeds for reproducibility
   torch.manual_seed(0)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True
   random.seed(0)
   np.random.seed(0)

   # Load the LLM output from the file
   with open(llm_output_path, "r") as f:
       text = f.read().strip()

   # Load packages
   from models import *
   from utils import *
   from text_utils import TextCleaner
   from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
   from Utils.PLBERT.util import load_plbert

   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   textcleaner = TextCleaner()

   # Load phonemizer
   global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')

   # Load configuration and models
   config_path = os.path.join("Models/LJSpeech/config.yml")
   with open(config_path, "r") as f:
       config = yaml.safe_load(f)

   ASR_config = config.get('ASR_config', False)
   ASR_path = config.get('ASR_path', False)
   text_aligner = load_ASR_models(ASR_path, ASR_config)

   F0_path = config.get('F0_path', False)
   pitch_extractor = load_F0_models(F0_path)

   BERT_path = config.get('PLBERT_dir', False)
   plbert = load_plbert(BERT_path)

   model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
   _ = [model[key].eval() for key in model]
   _ = [model[key].to(device) for key in model]

   params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
   params = params_whole['net']

   for key in model:
       if key in params:
           print('%s loaded' % key)
           try:
               model[key].load_state_dict(params[key])
           except:
               from collections import OrderedDict
               state_dict = params[key]
               new_state_dict = OrderedDict()
               for k, v in state_dict.items():
                   name = k[7:]  # remove `module.`
                   new_state_dict[name] = v
               model[key].load_state_dict(new_state_dict, strict=False)
   _ = [model[key].eval() for key in model]

   sampler = DiffusionSampler(
       model.diffusion.diffusion,
       sampler=ADPM2Sampler(),
       sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
       clamp=False
   )

   def length_to_mask(lengths):
       mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
       mask = torch.gt(mask + 1, lengths.unsqueeze(1))
       return mask

   to_mel = torchaudio.transforms.MelSpectrogram(
       n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
   mean, std = -4, 4

   def preprocess(wave):
       wave_tensor = torch.from_numpy(wave).float()
       mel_tensor = to_mel(wave_tensor)
       mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
       return mel_tensor

   def inference(text, noise, diffusion_steps=5, embedding_scale=1):
       text = text.strip().replace('"', '')
       ps = global_phonemizer.phonemize([text])
       ps = word_tokenize(ps[0])
       ps = ' '.join(ps)

       tokens = textcleaner(ps)
       tokens.insert(0, 0)
       tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

       with torch.no_grad():
           input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
           text_mask = length_to_mask(input_lengths).to(tokens.device)

           t_en = model.text_encoder(tokens, input_lengths, text_mask)
           bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
           d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

           s_pred = sampler(noise, embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps, embedding_scale=embedding_scale).squeeze(0)

           s = s_pred[:, 128:]
           ref = s_pred[:, :128]

           d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

           x, _ = model.predictor.lstm(d)
           duration = model.predictor.duration_proj(x)
           duration = torch.sigmoid(duration).sum(axis=-1)
           pred_dur = torch.round(duration.squeeze()).clamp(min=1)

           pred_dur[-1] += 5

           pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
           c_frame = 0
           for i in range(pred_aln_trg.size(0)):
               pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
               c_frame += int(pred_dur[i].data)

           en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
           F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
           out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)), F0_pred, N_pred, ref.squeeze().unsqueeze(0))

       return out.squeeze().cpu().numpy()

   def save_audio(audio_data, file_path):
       torchaudio.save(file_path, torch.tensor(audio_data).unsqueeze(0), 24000)

   # Run the inference
   noise = torch.randn(1, 1, 256).to(device)
   start = time.time()
   wav = inference(text, noise, diffusion_steps=10, embedding_scale=1)
   rtf = (time.time() - start) / (len(wav) / 24000)
   print(f"RTF = {rtf:5f}")
   save_audio(wav, audio_filename)
   print(f"Audio saved to {audio_filename}")
   ```

7. **Modify the file `MuseTalk/run_inference.py`:**
   ```python
   import argparse
   import os
   from omegaconf import OmegaConf
   import numpy as np
   import cv2
   import torch
   import glob
   import pickle
   import sys
   from tqdm import tqdm
   import copy
   import json
   from musetalk.utils.utils import get_file_type, get_video_fps, datagen
   from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
   from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending
   from musetalk.utils.utils import load_all_model
   import shutil

   import threading
   import queue

   import time

   # load model weights
   audio_processor, vae, unet, pe = load_all_model()
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   timesteps = torch.tensor([0], device=device)
   pe = pe.half()
   vae.vae = vae.vae.half()
   unet.model = unet.model.half()

   def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
       cap = cv2.VideoCapture(vid_path)
       count = 0
       while True:
           if count > cut_frame:
               break
           ret, frame = cap.read()
           if ret:
               cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
               count += 1
           else:
               break

   def osmakedirs(path_list):
       for path in path_list:
           os.makedirs(path) if not os.path.exists(path) else None


   @torch.no_grad()
   class Avatar:
       def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
           self.avatar_id = avatar_id
           self.video_path = video_path
           self.bbox_shift = bbox_shift
           self.avatar_path = f"./results/avatars/{avatar_id}"
           self.full_imgs_path = f"{self.avatar_path}/full_imgs"
           self.coords_path = f"{self.avatar_path}/coords.pkl"
           self.latents_out_path= f"{self.avatar_path}/latents.pt"
           self.video_out_path = f"{self.avatar_path}/vid_output/"
           self.mask_out_path =f"{self.avatar_path}/mask"
           self.mask_coords_path =f"{self.avatar_path}/mask_coords.pkl"
           self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
           self.avatar_info = {
               "avatar_id":avatar_id,
               "video_path":video_path,
               "bbox_shift":bbox_shift
           }
           self.preparation = preparation
           self.batch_size = batch_size
           self.idx = 0
           self.init()

       def init(self):
           if self.preparation:
               if os.path.exists(self.avatar_path):
                   # If the avatar already exists, do not recreate it, just load the existing data
                   print(f"{self.avatar_id} already exists, proceeding with the existing avatar.")
                   self.input_latent_list_cycle = torch.load(self.latents_out_path)
                   with open(self.coords_path, 'rb') as f:
                       self.coord_list_cycle = pickle.load(f)
                   input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                   input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                   self.frame_list_cycle = read_imgs(input_img_list)
                   with open(self.mask_coords_path, 'rb') as f:
                       self.mask_coords_list_cycle = pickle.load(f)
                   input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                   input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                   self.mask_list_cycle = read_imgs(input_mask_list)
               else:
                   print("*********************************")
                   print(f"  creating avator: {self.avatar_id}")
                   print("*********************************")
                   osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
                   self.prepare_material()
           else:
               if not os.path.exists(self.avatar_path):
                   print(f"{self.avatar_id} does not exist, you should set preparation to True")
                   sys.exit()

               with open(self.avatar_info_path, "r") as f:
                   avatar_info = json.load(f)

               if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                   response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                   if response.lower() == "c":
                       shutil.rmtree(self.avatar_path)
                       print("*********************************")
                       print(f"  creating avator: {self.avatar_id}")
                       print("*********************************")
                       osmakedirs([self.avatar_path,self.full_imgs_path,self.video_out_path,self.mask_out_path])
                       self.prepare_material()
                   else:
                       sys.exit()
               else:
                   self.input_latent_list_cycle = torch.load(self.latents_out_path)
                   with open(self.coords_path, 'rb') as f:
                       self.coord_list_cycle = pickle.load(f)
                   input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                   input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                   self.frame_list_cycle = read_imgs(input_img_list)
                   with open(self.mask_coords_path, 'rb') as f:
                       self.mask_coords_list_cycle = pickle.load(f)
                   input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                   input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                   self.mask_list_cycle = read_imgs(input_mask_list)


       def prepare_material(self):
           print("preparing data materials ... ...")
           with open(self.avatar_info_path, "w") as f:
               json.dump(self.avatar_info, f)

           if os.path.isfile(self.video_path):
               video2imgs(self.video_path, self.full_imgs_path, ext = 'png')
           else:
               print(f"copy files in {self.video_path}")
               files = os.listdir(self.video_path)
               files.sort()
               files = [file for file in files if file.split(".")[-1]=="png"]
               for filename in files:
                   shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
           input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

           print("extracting landmarks...")
           coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
           input_latent_list = []
           idx = -1
           # maker if the bbox is not sufficient
           coord_placeholder = (0.0,0.0,0.0,0.0)
           for bbox, frame in zip(coord_list, frame_list):
               idx = idx + 1
               if bbox == coord_placeholder:
                   continue
               x1, y1, x2, y2 = bbox
               crop_frame = frame[y1:y2, x1:x2]
               resized_crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
               latents = vae.get_latents_for_unet(resized_crop_frame)
               input_latent_list.append(latents)

           self.frame_list_cycle = frame_list + frame_list[::-1]
           self.coord_list_cycle = coord_list + coord_list[::-1]
           self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
           self.mask_coords_list_cycle = []
           self.mask_list_cycle = []

           for i,frame in enumerate(tqdm(self.frame_list_cycle)):
               cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png",frame)

               face_box = self.coord_list_cycle[i]
               mask,crop_box = get_image_prepare_material(frame,face_box)
               cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png",mask)
               self.mask_coords_list_cycle += [crop_box]
               self.mask_list_cycle.append(mask)

           with open(self.mask_coords_path, 'wb') as f:
               pickle.dump(self.mask_coords_list_cycle, f)

           with open(self.coords_path, 'wb') as f:
               pickle.dump(self.coord_list_cycle, f)

           torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))
           #

       def process_frames(self,
                          res_frame_queue,
                          video_len,
                          skip_save_images):
           print(video_len)
           while True:
               if self.idx>=video_len-1:
                   break
               try:
                   start = time.time()
                   res_frame = res_frame_queue.get(block=True, timeout=1)
               except queue.Empty:
                   continue

               bbox = self.coord_list_cycle[self.idx%(len(self.coord_list_cycle))]
               ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx%(len(self.frame_list_cycle))])
               x1, y1, x2, y2 = bbox
               try:
                   res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
               except:
                   continue
               mask = self.mask_list_cycle[self.idx%(len(self.mask_list_cycle))]
               mask_crop_box = self.mask_coords_list_cycle[self.idx%(len(self.mask_coords_list_cycle))]
               #combine_frame = get_image(ori_frame,res_frame,bbox)
               combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)

               if skip_save_images is False:
                   cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png",combine_frame)
               self.idx = self.idx + 1

       def inference(self,
                     audio_path,
                     out_vid_name,
                     fps,
                     skip_save_images):
           os.makedirs(self.avatar_path+'/tmp',exist_ok =True)
           print("start inference")
           ############################################## extract audio feature ##############################################
           start_time = time.time()
           whisper_feature = audio_processor.audio2feat(audio_path)
           whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
           print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
           ############################################## inference batch by batch ##############################################
           video_num = len(whisper_chunks)
           res_frame_queue = queue.Queue()
           self.idx = 0
           # # Create a sub-thread and start it
           process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, skip_save_images))
           process_thread.start()

           gen = datagen(whisper_chunks,
                         self.input_latent_list_cycle,
                         self.batch_size)
           start_time = time.time()
           res_frame_list = []

           for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/self.batch_size)))):
               audio_feature_batch = torch.from_numpy(whisper_batch)
               audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                            dtype=unet.model.dtype)
               audio_feature_batch = pe(audio_feature_batch)
               latent_batch = latent_batch.to(dtype=unet.model.dtype)

               pred_latents = unet.model(latent_batch,
                                         timesteps,
                                         encoder_hidden_states=audio_feature_batch).sample
               recon = vae.decode_latents(pred_latents)
               for res_frame in recon:
                   res_frame_queue.put(res_frame)
           # Close the queue and sub-thread after all tasks are completed
           process_thread.join()

           if args.skip_save_images is True:
               print('Total process time of {} frames without saving images = {}s'.format(
                           video_num,
                           time.time()-start_time))
           else:
               print('Total process time of {} frames including saving images = {}s'.format(
                           video_num,
                           time.time()-start_time))

           if out_vid_name is not None and args.skip_save_images is False:
               # optional
               cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
               print(cmd_img2video)
               os.system(cmd_img2video)

               output_vid = os.path.join(self.video_out_path, out_vid_name+".mp4") # on
               cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
               print(cmd_combine_audio)
               os.system(cmd_combine_audio)

               os.remove(f"{self.avatar_path}/temp.mp4")
               shutil.rmtree(f"{self.avatar_path}/tmp")
               print(f"result is save to {output_vid}")


   if __name__ == "__main__":
       '''
       This script is used to simulate online chatting and applies necessary pre-processing such as face detection and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.
       '''

       parser = argparse.ArgumentParser()
       parser.add_argument("--inference_config",
                           type=str,
                           default="configs/inference/realtime.yaml",
       )
       parser.add_argument("--fps",
                           type=int,
                           default=25,
       )
       parser.add_argument("--batch_size",
                           type=int,
                           default=4,
       )
       parser.add_argument("--skip_save_images",
                           action="store_true",
                           help="Whether skip saving images for better generation speed calculation",
       )

       args = parser.parse_args()

       inference_config = OmegaConf.load(args.inference_config)
       print(inference_config)


       for avatar_id in inference_config:
           data_preparation = inference_config[avatar_id]["preparation"]
           video_path = inference_config[avatar_id]["video_path"]
           bbox_shift = inference_config[avatar_id]["bbox_shift"]
           avatar = Avatar(
               avatar_id = avatar_id,
               video_path = video_path,
               bbox_shift = bbox_shift,
               batch_size = args.batch_size,
               preparation= data_preparation)

           audio_clips = inference_config[avatar_id]["audio_clips"]
           for audio_num, audio_path in audio_clips.items():
               print("Inferring using:",audio_path)
               avatar.inference(audio_path,
                                audio_num,
                                args.fps,
                                args.skip_save_images)
   ```

## Inference

1. **Change directory to `Human_TalkingHeads`:**
   ```sh
   cd Human_TalkingHeads
   ```

2. **Upload your custom images in the `media/pics` folder.**
3. **Upload the motion transferred videos generated beforehand using the "Simple_Pipeline" provided to the `media/motion_video` folder.**
4. **Run the following commands:**
   ```sh
   python manage.py makemigrations
   python manage.py migrate
   python manage.py runserver 7000
   ```
   This hosts the server on port 7000.

**Note**: Make sure to delete the existing avatar before trying it on a differnt person.
Please refer to the provided documentation if you face any issues.
