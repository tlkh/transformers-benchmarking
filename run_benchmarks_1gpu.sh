#!/usr/bin/env bash

# select first GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

export EXP_PREFIX="results/ec2-v100/amp"
export MODEL_NAME="distilroberta-base"
export BATCH_SIZE=8
export SEQ_LEN=128
export EXP_NAME="$EXP_PREFIX/$MODEL_NAME-$SEQ_LEN-$BATCH_SIZE"

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

export BATCH_SIZE=32
export SEQ_LEN=128
export EXP_NAME="$EXP_PREFIX-$MODEL_NAME-$SEQ_LEN-$BATCH_SIZE"

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

export BATCH_SIZE=64
export SEQ_LEN=128
export EXP_NAME="$EXP_PREFIX-$MODEL_NAME-$SEQ_LEN-$BATCH_SIZE"

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
 
export BATCH_SIZE=128
export SEQ_LEN=128
export EXP_NAME="$EXP_PREFIX-$MODEL_NAME-$SEQ_LEN-$BATCH_SIZE"

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

export BATCH_SIZE=8
export SEQ_LEN=512
export EXP_NAME="$EXP_PREFIX-$MODEL_NAME-$SEQ_LEN-$BATCH_SIZE"

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

export BATCH_SIZE=32
export SEQ_LEN=512
export EXP_NAME="$EXP_PREFIX-$MODEL_NAME-$SEQ_LEN-$BATCH_SIZE"

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

export BATCH_SIZE=64
export SEQ_LEN=512
export EXP_NAME="$EXP_PREFIX-$MODEL_NAME-$SEQ_LEN-$BATCH_SIZE"

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

export BATCH_SIZE=128
export SEQ_LEN=512
export EXP_NAME="$EXP_PREFIX-$MODEL_NAME-$SEQ_LEN-$BATCH_SIZE"

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

  
