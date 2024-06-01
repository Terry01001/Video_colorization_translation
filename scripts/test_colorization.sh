set -ex

python test.py --dataroot ./datasets/colorization --name color_pix2pix --model colorization --dataset_mode colorization --input_nc 1 --output_nc 2

