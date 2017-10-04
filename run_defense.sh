#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR OUTPUT_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_FILE - file to store classification labels
#

INPUT_DIR=$1
OUTPUT_FILE=$2
CHECKPOINT1="retrained_network0.ckpt"
CHECKPOINT2="retrained_network2.ckpt"
CHECKPOINT3="retrained_network3.ckpt"
CHECKPOINT4="retrained_network4.ckpt"
CHECKPOINT5="inception_v3.ckpt"

python ensemble.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --ckpt1=${CHECKPOINT1} \
  --ckpt2=${CHECKPOINT2} \
  --ckpt3=${CHECKPOINT3} \
  --ckpt4=${CHECKPOINT4} \
  --ckpt5=${CHECKPOINT5} \
  --weight1=0.23 \
  --weight2=0.21 \
  --weight3=0.20 \
  --weight4=0.18 \
  --weight5=0.18 \
