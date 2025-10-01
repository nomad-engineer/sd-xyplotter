
## CONTEXT.md

# SDXL XY Plot Generator - Technical Specification

## Overview

A Python-based tool that generates comparison grids (XY plots) for Stable Diffusion XL image generation. The system allows users to vary any two parameters across axes to create visual comparisons, such as comparing different models, LoRAs, prompts, or generation settings.

## Core Functionality

### Primary Purpose
Generate a grid of images where:
- X-axis (horizontal) represents variations of one parameter
- Y-axis (vertical) represents variations of another parameter
- Each cell contains an image generated with that specific combination
- All other parameters remain constant across the grid

### Key Design Principles
1. **Flexibility**: Any parameter that can vary should be plottable on an axis
2. **Simplicity**: Single values are constants, multiple values can be axes
3. **Efficiency**: Minimize model reloading and optimize memory usage
4. **Compatibility**: Support standard SDXL models, LoRAs, and generation parameters

## Input Specifications

### Configuration File (TOML)
Primary input that defines the entire generation job.

#### Structure
```
[xy_plot]
- x_axis: string - parameter name for horizontal axis
- y_axis: string - parameter name for vertical axis
- x_axis_label: string (optional) - display name for X axis
- y_axis_label: string (optional) - display name for Y axis

[checkpoint]
- path: string - file path or directory path
- values: array (alternative) - list of checkpoint paths

[prompt]
- values: array - list of prompt strings
- values_file: string (alternative) - path to text file with prompts

[lora1], [lora2], etc.
- path: string - file or directory path
- weight: float - strength multiplier (default 1.0)
- values: array (alternative) - list of LoRA paths

[lora1_weights], [lora2_weights], etc.
- path: string - single LoRA file
- weights: array - list of weight values to test

[generation]
- negative_prompt: string
- seed: integer (-1 for random)
- resolution: integer (square images)
- steps: integer
- cfg_scale: float
- scheduler: string
- clip_skip: integer

[output]
- output_dir: string - where to save results
- save_individual_images: boolean
- individual_images_dir: string
- grid_image_quality: integer (1-100)
- create_preview: boolean
- preview_scale: float (0-1)

[grid]
- margin_size: integer - pixels for labels
- label_font_size: integer
- label_font_color: string (hex color)
- label_line_spacing: integer
- max_label_chars: integer
- grid_line_width: integer
- grid_line_color: string (hex color)
- background_color: string (hex color)

[memory_optimization]
- enable_cpu_offload: boolean
- enable_attention_slicing: boolean
- enable_vae_tiling: boolean
```

#### Parameter Rules
1. **Single Value**: Parameter is constant for all generations
2. **Multiple Values**: Parameter can be used as an axis variable
3. **Maximum 2 Variable Parameters**: Only X and Y axes can vary
4. **Directory Paths**: Automatically expanded to find all .safetensors files
5. **Naming Convention**: 
   - `lora1`, `lora2`: Different LoRA models
   - `lora1_weights`: Weight variations for a specific LoRA

### Prompt File Format
Text file with one prompt per line, supporting:
- Plain text prompts
- Comments (lines starting with # or //)
- Kohya-style flags (automatically stripped):
  - `--n`, `--negative`: negative prompt (ignored)
  - `--steps`, `--s`: step count (ignored)
  - `--cfg`, `--cfg-scale`: CFG scale (ignored)
  - `--seed`: random seed (ignored)
  - `--w`, `--width`, `--h`, `--height`: dimensions (ignored)
  - Other flags are also removed

### Command Line Interface
```
python main.py [options]

Options:
--config PATH: Configuration file path (default: config.toml)
--set PARAM=VALUE: Override configuration values
--dry-run: Show configuration without generating
--save-config PATH: Save modified configuration
```
