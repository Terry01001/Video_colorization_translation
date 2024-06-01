#!/bin/bash
set -ex

FILENAMES=("Beatles.mp4" "Chaplin_Factory.mp4" "Chaplin_Barber.mp4")

for FILENAME in "${FILENAMES[@]}"
do
    BASENAME="${FILENAME%.*}"
    python test.py --dataroot ./datasets/colorization --name color_pix2pix --model videocolorization --dataset_mode videocolorization  --video_name $FILENAME --input_nc 1 --output_nc 2

    NAMES=("style_monet_pretrained" "style_cezanne_pretrained" "style_ukiyoe_pretrained" "style_vangogh_pretrained")

    for NAME in "${NAMES[@]}"
    do
        echo "Running experiment for $NAME"
        python test.py --dataroot ./results/color_pix2pix/$BASENAME --name $NAME --phase test --no_dropout
    done

done

# python test.py --dataroot ./datasets/colorization --name color_pix2pix --model videocolorization --dataset_mode videocolorization  --video_name "Beatles.mp4" --input_nc 1 --output_nc 2






