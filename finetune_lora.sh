#!/bin/bash

export NCCL_P2P_DISABLE=1

MODEL_VERSION="llava-v1.6-mistral-7b"
FINETUNE_VERSION="alfworld-gpt4-45k"

deepspeed --include localhost:0,1,2 \
    LLaVA/llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path liuhaotian/llava-v1.6-mistral-7b \
    --version v1 \
    --data_path ./sft-data/$FINETUNE_VERSION.json \
    --image_folder ./sft-data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$MODEL_VERSION-$FINETUNE_VERSION-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --run_name "$MODEL_VERSION-$FINETUNE_VERSION-lora" \
    --report_to wandb 
