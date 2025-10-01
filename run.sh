#!/bin/bash
source venv/bin/activate

python main.py \
  --set "checkpoint.path=/home/adam/SanDisk/Files/AI/sd/checkpoints/illustrious/plantMilkModelSuite_hempII.safetensors" \
  --set "lora1.path=/home/adam/Apps/runpod/jobs/fatimmobile_v9_finetune_hard/lora_nr32" \
  --set "prompt.values_file=prompts.txt"