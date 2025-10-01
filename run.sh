#!/bin/bash
source venv/bin/activate

# Basic usage
#python main.py

# Override config file
#python main.py --config my_config.toml

# Override specific settings
#python main.py --model-dir /path/to/models --output-dir ./my_output

# For LoRA mode
python main.py \
    --base-checkpoint "/home/adam/SanDisk/Files/AI/sd/checkpoints/illustrious/plantMilkModelSuite_hempII.safetensors" \
    --model-dir "/home/adam/Apps/runpod/jobs/fatimmobile_v9_finetune_hard/lora_nr32"

#python main.py \
#    --model-dir "/home/adam/Apps/runpod/jobs/fatimmobile_v9_finetune_hard/checkpoint" 
