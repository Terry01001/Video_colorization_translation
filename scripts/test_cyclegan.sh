#!/bin/bash
set -ex

NAMES=("style_monet_pretrained" "style_cezanne_pretrained" "style_ukiyoe_pretrained" "style_vangogh_pretrained")

for NAME in "${NAMES[@]}"
do
    echo "Running experiment for $NAME"
    python test.py --dataroot ./results/color_pix2pix/test_latest/images --name $NAME --phase test --no_dropout
done
