#!/usr/bin/env bash

# select first GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

export EXP_PREFIX="results/v100-16gb-300w/amp"

for MODEL_NAME in "distilroberta-base" "roberta-base" "roberta-large"
do

  for BATCH_SIZE in 1 8 32 64 128 256
  do

    for SEQ_LEN in 128 256 512
    do

      export EXP_NAME="$EXP_PREFIX/$MODEL_NAME/$SEQ_LEN-$BATCH_SIZE"
      # profile
      dlprof --mode=simple --reports=summary --output_path=$EXP_NAME \
        --delay=10 --force=true --formats=json --suppress_tb_files=true \
        python run_training.py --profile_mode \
          --amp --batch_size=$BATCH_SIZE --max_seq_len=$SEQ_LEN \
          --model_name=$MODEL_NAME --exp_name=$EXP_NAME 
      # benchmark
      python run_training.py \
        --amp --batch_size=$BATCH_SIZE --max_seq_len=$SEQ_LEN \
        --model_name=$MODEL_NAME --exp_name=$EXP_NAME
      rm $EXP_NAME/nsys_profile*

    done

  done

done
