
# VASA-1 Implementation

This repository is a liberal implementation of the VASA-1 paper, coding the architectures clearly mentioned and cherry-picking the codes of architectures from other research implementations that are vaguely presented in the VASA-1 and Megaportraits papers.

## Prerequisites for Installation
- Install Anaconda.
- Install git.

## Installation

1. **Create and activate an environment:**
   ```sh
   conda create -n env
   conda activate env
   ```

2. **Install Python and git:**
   ```sh
   conda install python
   conda install git
   ```

3. **Clone this branch and change the directory to `VASA-1` or download the zip file from this branch:**
   ```sh
   git clone -b VASA-1 https://github.com/vishwaraghava009/IBY_sumbission_Vishwa_Raghava.git
   cd VASA-1
   ```

4. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

## Configuration and Training

1. **Modify the `configs/training/train_config.yaml` file:**
   - Change the paths of `video_dir:` to point to the Large Video Set to train upon.
   - Update `json_file:` to point to the metadata file of the video dataset used.

2. **Adjust training parameters:**
   - Modify `batch_size`, `base_epochs`, `save_interval`, `log_interval`, etc. according to your convenience.

3. **Train the base model:**
   ```sh
   python train.py
   ```

4. **Train the High-resolution and student models:**
   ```sh
   python train_full.py
   ```

5. **Train or work on VASA-1 training:**
   ```sh
   python train_vasa.py
   ```

## CelebV-HQ and FFHQ Datasets

### CelebV-HQ Dataset
To download the CelebV-HQ dataset, please use the following academic torrent:
- [CelebV-HQ Academic Torrent](https://academictorrents.com/details/843b5adb0358124d388c4e9836654c246b988ff4)
- This dataset can be used to train base and student models.

### FFHQ Dataset
To download the FFHQ dataset, please use the following academic torrent:
- [FFHQ Academic Torrent](https://academictorrents.com/details/1c1e60f484e911b564de6b4d8b643e19154d5809)
- This dataset can be used for the High-resolution model.

---

By following the steps outlined above, you will be able to set up, configure, and train the VASA-1 implementation effectively. For any further assistance or detailed explanations, please refer to the documentation provided within the repository.
