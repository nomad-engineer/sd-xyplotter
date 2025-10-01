# SDXL XY Plot Generator

A flexible tool for generating XY comparison grids with Stable Diffusion XL models. Create visual comparisons of different parameters like checkpoints, LoRAs, prompts, and generation settings in a grid format.

## Features

- **Flexible Axis Configuration**: Plot any two parameters against each other
- **Multiple Model Support**: Compare checkpoints, LoRAs, or both
- **Batch Processing**: Generate entire grids efficiently with memory optimization
- **Kohya Format Support**: Load prompts from Kohya-formatted text files
- **CLI Overrides**: Modify any configuration value from the command line
- **Memory Optimization**: Automatic VRAM management for GPUs with limited memory

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have CUDA-capable GPU (or CPU mode will be used automatically)

## Quick Start

1. Create a `config.toml` file (see Configuration section)
2. Run the generator:
```bash
python main.py --config config.toml
```

## Configuration

The configuration uses TOML format with the following structure:

### Basic Structure

```toml
[xy_plot]
x_axis = "prompt"      # What varies horizontally
y_axis = "checkpoint"  # What varies vertically

[checkpoint]
# Single file = constant, used as base for all generations
path = "/path/to/model.safetensors"
# OR directory = multiple files for axis variation
path = "/path/to/models/"
# OR explicit list
values = ["/path/to/model1.safetensors", "/path/to/model2.safetensors"]

[prompt]
# From file (supports Kohya format)
values_file = "prompts.txt"
# OR inline list
values = ["prompt 1", "prompt 2", "prompt 3"]

[generation]
negative_prompt = "low quality, blurry"
seed = 42
resolution = 1024
steps = 30
cfg_scale = 7.0
```

### Multi-Parameter Rules

**Key Concept**: Any parameter with multiple values can be used as an axis. Parameters with single values are constants applied to all generations.

#### Example 1: Checkpoints vs Prompts
```toml
[xy_plot]
x_axis = "prompt"
y_axis = "checkpoint"

[checkpoint]
# Multiple values = can be used as axis
values = [
    "/models/anime.safetensors",
    "/models/realistic.safetensors",
    "/models/artistic.safetensors"
]

[prompt]
values = ["portrait", "landscape", "abstract art"]

# Result: 3x3 grid (3 checkpoints × 3 prompts = 9 images)
```

#### Example 2: LoRA Weights vs Prompts
```toml
[xy_plot]
x_axis = "prompt"
y_axis = "lora1_weights"

[checkpoint]
# Single value = base model (constant)
path = "/models/base_sdxl.safetensors"

[lora1_weights]
path = "/loras/style.safetensors"
weights = [0.5, 0.75, 1.0, 1.25]  # Multiple weights

[prompt]
values = ["anime style", "realistic", "oil painting"]

# Result: 3x4 grid (3 prompts × 4 weights = 12 images)
```

#### Example 3: Two LoRAs with Different Weights
```toml
[xy_plot]
x_axis = "lora1_weights"
y_axis = "lora2_weights"

[checkpoint]
path = "/models/base.safetensors"  # Base model

[lora1_weights]
path = "/loras/style.safetensors"
weights = [0.5, 1.0, 1.5]

[lora2_weights]
path = "/loras/character.safetensors"
weights = [0.5, 1.0, 1.5]

[generation]
prompt = "portrait, masterpiece"  # Fixed prompt

# Result: 3x3 grid testing LoRA combinations
```

### Parameter Types

#### Checkpoints
```toml
[checkpoint]
# Single file (constant)
path = "/path/to/model.safetensors"

# Directory (finds all .safetensors files)
path = "/path/to/models/"

# Explicit list
values = ["/path/to/model1.safetensors", "/path/to/model2.safetensors"]
```

#### LoRAs
```toml
[lora1]  # or [lora2], [lora3], etc.
# Single LoRA
path = "/path/to/lora.safetensors"
weight = 1.0

# Directory of LoRAs
path = "/path/to/loras/"
weight = 0.8  # Applied to all

# Multiple weights for same LoRA
[lora1_weights]
path = "/path/to/lora.safetensors"
weights = [0.5, 0.75, 1.0, 1.25]
```

#### Prompts
```toml
[prompt]
# From file (supports Kohya format with --n, --steps, etc.)
values_file = "prompts.txt"

# Inline list
values = [
    "beautiful landscape",
    "portrait photography",
    "abstract art"
]
```

#### Generation Parameters
```toml
[resolution]
values = [512, 768, 1024]

[steps]
values = [20, 30, 40, 50]

[cfg_scale]
values = [5.0, 7.0, 9.0, 11.0]

[seed]
values = [42, 123, 456, 789]
```

### Valid Axis Combinations

You can plot any two variable parameters:
- `checkpoint` vs `prompt`
- `checkpoint` vs `lora1`
- `lora1` vs `lora2`
- `prompt` vs `cfg_scale`
- `steps` vs `resolution`
- `lora1_weights` vs `prompt`
- etc.

**Important**: Maximum 2 parameters can have multiple values (one for each axis).

## Prompt Files

Create a `prompts.txt` file with one prompt per line:

```txt
# Comments are ignored
a beautiful anime girl, masterpiece --n low quality --steps 30 --cfg 7.5
landscape photography, mountains --negative ugly --cfg-scale 8.0
cyberpunk city at night --w 1024 --h 1024

# Kohya-style flags are automatically removed:
# --n, --negative: negative prompt
# --s, --steps: sampling steps
# --cfg, --cfg-scale: CFG scale
# --seed: random seed
# --w, --width, --h, --height: dimensions
```

## Command Line Usage

### Basic Usage
```bash
python main.py --config config.toml
```

### Override Configuration
```bash
# Change single values
python main.py --set checkpoint.path=/path/to/model.safetensors

# Change axis assignment
python main.py --set xy_plot.x_axis=checkpoint --set xy_plot.y_axis=prompt

# Override generation parameters
python main.py --set generation.steps=50 --set generation.cfg_scale=8.5

# Use different prompt file
python main.py --set prompt.values_file=other_prompts.txt

# Override with lists
python main.py --set "checkpoint.values=[/model1.safetensors,/model2.safetensors]"
```

### Utility Commands
```bash
# Preview configuration without generating
python main.py --dry-run

# Save modified configuration
python main.py --set generation.steps=50 --save-config my_config.toml

# Multiple overrides
python main.py \
  --set checkpoint.path=/models/new_model.safetensors \
  --set generation.steps=40 \
  --set output.output_dir=./results
```

## Output

The generator creates:
- **XY Grid Image**: Full resolution grid with all combinations
- **Preview Image**: Smaller preview (25% by default)
- **Individual Images**: Each generated image (optional)
- **Metadata JSON**: Complete configuration and parameters used

Output structure:
```
output/
├── xy_plot_20240101_120000.png        # Main grid
├── xy_plot_20240101_120000_preview.png # Preview
├── xy_plot_20240101_120000_metadata.json
└── individual_20240101_120000/        # Individual images (optional)
    ├── y00_x00_model1_prompt1.png
    ├── y00_x01_model1_prompt2.png
    └── ...
```

## Memory Optimization

For GPUs with limited VRAM:

```toml
[memory_optimization]
enable_cpu_offload = true      # Move models between CPU/GPU
enable_attention_slicing = true # Reduce attention memory usage
enable_vae_tiling = true       # Decode images in tiles

[generation]
resolution = 768  # Reduce if still running out of memory
```

## Tips

1. **Start Small**: Test with 2-3 values per axis before large grids
2. **Use Preview**: Check the preview image before opening large grids
3. **Save Configurations**: Use `--save-config` to save successful setups
4. **Monitor VRAM**: The tool shows GPU memory and auto-enables optimizations
5. **Batch Similar Models**: Group similar checkpoints/LoRAs for better comparison

## Troubleshooting

**Out of Memory**: 
- Enable memory optimizations in config
- Reduce resolution
- Use fewer steps
- Process fewer models at once

**LoRA Not Loading**:
- Ensure base checkpoint is specified
- Check file paths are correct
- Verify LoRA is compatible with the base model

**Slow Generation**:
- Disable CPU offload if you have enough VRAM
- Reduce resolution or steps
- Use fewer complex LoRAs

## Examples

See the `examples/` directory for complete configuration examples for common use cases.


Override format:
- Simple: `generation.steps=50`
- Lists: `prompt.values=[val1,val2,val3]`
- Strings: `checkpoint.path=/path/to/file`
- Booleans: `memory_optimization.enable_cpu_offload=false`

## Processing Logic

### Parameter Resolution
1. Load base configuration from TOML
2. Apply CLI overrides
3. Identify variable parameters (those with multiple values)
4. Validate exactly 2 parameters are variable
5. Assign parameters to X and Y axes

### Generation Flow
1. **Initialize Generator**
   - Load required libraries (diffusers, transformers)
   - Detect GPU capabilities
   - Apply memory optimizations

2. **Parameter Cartesian Product**
   - Create all combinations of X and Y values
   - Each combination inherits constant parameters
   - Total images = len(X values) × len(Y values)

3. **Model Management**
   - Checkpoints: Load when changed, keep in memory if unchanged
   - LoRAs: Apply on top of current checkpoint
   - Multiple LoRAs: Can stack (lora1 + lora2)
   - Memory clearing between major model changes

4. **Image Generation**
   For each cell (x, y):
   - Load/apply required models
   - Generate image with combined parameters
   - Handle failures with placeholder images
   - Save individual images if requested

5. **Grid Assembly**
   - Create canvas with margins for labels
   - Draw wrapped text labels for both axes
   - Paste generated images in grid positions
   - Add grid lines and axis titles

## Output Specifications

### Files Created
1. **Main Grid Image** (`xy_plot_TIMESTAMP.png`)
   - Full resolution grid
   - Labeled axes with text wrapping
   - Grid lines for clarity
   - All images at specified resolution

2. **Preview Image** (`xy_plot_TIMESTAMP_preview.png`)
   - Scaled down version (default 25%)
   - Same layout as main grid
   - Quick viewing for large grids

3. **Metadata File** (`xy_plot_TIMESTAMP_metadata.json`)
   ```json
   {
     "timestamp": "YYYYMMDD_HHMMSS",
     "x_axis": "parameter_name",
     "y_axis": "parameter_name",
     "x_values": ["value1", "value2"],
     "y_values": ["value1", "value2"],
     "config": {full configuration object}
   }
   ```

4. **Individual Images** (optional, in subdirectory)
   - Named: `yYY_xXX_ylabel_xlabel.png`
   - Each with accompanying metadata JSON

### Grid Layout
```
        [Title X]
[Title] [Label1] [Label2] [Label3]
[Y1]    [Image]  [Image]  [Image]
[Y2]    [Image]  [Image]  [Image]
[Y3]    [Image]  [Image]  [Image]
```

## Model Handling

### Checkpoint Behavior
- Single checkpoint: Base model for all generations
- Multiple checkpoints: Can be axis variable
- Checkpoint switching: Full model reload required
- Memory: Previous checkpoint unloaded before loading new

### LoRA Behavior
- Requires base checkpoint
- Can apply multiple LoRAs simultaneously
- Weight parameter controls strength
- Can vary: LoRA selection, LoRA weights, or both
- Efficient: Only LoRA weights change, base model stays loaded

### Valid Combinations
- Checkpoint only (no LoRA)
- Checkpoint + Single LoRA
- Checkpoint + Multiple LoRAs
- Single Checkpoint + LoRA variations
- Multiple Checkpoints + Same LoRA
- Multiple Checkpoints + Multiple LoRAs

## Memory Management

### Automatic Optimizations
- Detect GPU VRAM on startup
- Enable CPU offload for <12GB VRAM
- Enable attention slicing for <16GB VRAM
- Enable VAE tiling for <12GB VRAM

### Manual Controls
- Configure optimizations in config file
- Reduce resolution for lower VRAM usage
- Adjust batch processing
- Force garbage collection between models

### Recovery Mechanisms
- Catch CUDA OOM errors
- Retry with reduced resolution
- Create placeholder for failed generations
- Continue with remaining images

## Extension Points

### Adding New Parameters
Any parameter that affects generation can be made variable:
1. Add parameter to configuration parsing
2. Define how it applies to generation
3. Handle single vs multiple values
4. Add to valid axis parameters

### Adding New Model Types
1. Define loading mechanism
2. Specify combination rules with existing models
3. Handle memory management
4. Update parameter parsing

### Custom Schedulers
- Add to SCHEDULERS dictionary
- Map configuration names to scheduler classes
- No other changes needed

### Output Formats
- Image saving uses PIL
- Can add different formats
- Metadata is JSON (could add CSV, etc.)

## Error Handling

### Validation Errors
- Missing configuration sections
- Invalid axis parameters
- Too many variable parameters
- File not found errors

### Generation Errors
- Model loading failures
- CUDA out of memory
- Invalid parameter combinations
- Corrupted model files

### Recovery Strategy
1. Log error with context
2. Create placeholder if possible
3. Continue with remaining images
4. Report summary at end

## Performance Considerations

### Optimization Strategies
- Minimize model reloading
- Keep base model in memory when varying LoRAs
- Process in optimal order (group by model)
- Clear memory proactively

### Bottlenecks
- Model loading (disk I/O)
- VRAM limitations
- CPU-GPU transfer (when offloading)
- Image encoding/decoding

### Scaling Limits
- Grid size limited by memory (both RGB and VRAM)
- Practical limit ~10x10 grid at 1024px
- Can generate larger grids at lower resolution
- Individual image saving allows unlimited grid size

## Future Enhancements

### Potential Features
1. **3D Plots**: Third parameter via multiple grids
2. **Animation**: Create GIFs from parameter sweeps  
3. **Parallel Generation**: Multiple GPUs support
4. **Smart Ordering**: Optimize generation order
5. **Partial Grids**: Resume interrupted generations
6. **Parameter Extraction**: Use prompts file parameters
7. **Automatic Optimal Settings**: Based on hardware
8. **Web Interface**: Browser-based configuration
9. **Batch Jobs**: Queue multiple grids
10. **Comparison Metrics**: Auto-calculate similarity scores

### Architecture Improvements
1. **Plugin System**: Modular parameter handlers
2. **Pipeline Abstraction**: Support other models (SD1.5, etc.)
3. **Configuration Inheritance**: Base configs with overrides
4. **Result Database**: Track all generations
5. **Distributed Processing**: Network-based generation

This specification provides complete information for reimplementation or enhancement while remaining implementation-agnostic.
```

These documents provide comprehensive user instructions and technical specifications for the SDXL XY Plot Generator, making it easy for users to understand the system and for developers to extend or reimplement it.