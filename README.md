<img src='imgs/Chaplin_Barber.gif' align="right" width=1000> 

<br><br><br>

# Video colorization and translation using pix2pix and CycleGAN


<img src='imgs/Overall_Architecture.jpg' width=480>

This project leverages the power of conditional GANs to colorize grayscale videos and CycleGAN for style transfer. The colorization process uses the Pix2Pix model, while the style transfer is accomplished with CycleGAN. The demo video can be viewed [here](https://drive.google.com/drive/folders/1SVA7trdCwTGiufNew1d3kUPHCUO1aqrz?usp=drive_link). Please refer to [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for more information.


## Project Overview

### Video Colorization
The video colorization process involves converting grayscale frames of a video into colorized frames using a Pix2Pix model. This model is trained on pairs of grayscale and color images to learn the mapping from grayscale to color.

### Style Transfer
Once the video frames are colorized, they can be further processed to transfer various artistic styles using CycleGAN. The CycleGAN model does not require paired images for training and can learn to translate an image from one domain (e.g., real photos) to another domain (e.g., Monet's paintings) with unpaired datasets.

### Workflow
1. **Colorize Grayscale Videos**: Using Pix2Pix, convert grayscale frames into color frames.
2. **Style Transfer on Colorized Frames**: Apply different artistic styles to the colorized frames using pretrained CycleGAN models.
3. **Video Reconstruction**: Combine the processed frames back into a video format.


## Prerequisites
- Linux 
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/Terry01001/Video_colorization_translation.git
cd Video_colorization_translation
```

- Environmental Setup
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

### Train colorization model (pix2pix based)
- You can train a colorization model with the following script:
```bash
bash ./scripts/train_colorization.sh
```
After training, the model is saved at `./checkpoints/color_pix2pix/latest_net_G.pth`.


### Apply pre-trained models (CycleGAN)
- You can download style_transfer pretrained models with the following instructions:
```bash
bash ./scripts/download_cyclegan_model.sh style_monet
bash ./scripts/download_cyclegan_model.sh style_cezanne
bash ./scripts/download_cyclegan_model.sh style_ukiyoe
bash ./scripts/download_cyclegan_model.sh style_vangogh
```
- The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`.

- Then generate the results using
```bash
bash ./scripts/test_video_colorization_translation.sh
```

## Acknowledgments
This project is based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), thanks to their hard work.

