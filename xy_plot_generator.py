"""
Flexible XY Plot Generator for SDXL
"""

import os
import sys
import torch
import toml
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
from tqdm.auto import tqdm
from dataclasses import dataclass

from sdxl_generator import SDXLGenerator


@dataclass
class PlotParameter:
    """Represents a parameter that can be varied in the plot"""
    name: str
    values: List[Any]
    display_names: List[str] = None
    is_variable: bool = True
    
    def __post_init__(self):
        if self.display_names is None:
            self.display_names = [str(v) for v in self.values]
        # Auto-detect if variable based on number of values
        if len(self.values) == 1:
            self.is_variable = False


class XYPlotConfig:
    """Parse and validate XY plot configuration"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = toml.load(f)
        
        self._validate_config()
        self._parse_parameters()
        self._parse_axes()
    
    def _load_prompts_from_file(self, file_path: str) -> List[str]:
        """Load prompts from a text file, ignoring Kohya-style flags"""
        prompts = []
        
        # Kohya flags to ignore (common ones)
        kohya_flags = {
            '--n', '--negative', '--neg',
            '--s', '--steps', '--sampling-steps',
            '--c', '--cfg', '--cfg-scale', '--scale',
            '--seed', '--width', '--height', '--w', '--h',
            '--sampler', '--scheduler', '--clip-skip',
            '--ar', '--aspect-ratio', '--batch', '--model'
        }
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip whitespace
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip comment lines
                if line.startswith('#') or line.startswith('//'):
                    continue
                
                # Remove Kohya-style flags and their values
                # This handles formats like: "prompt text --n negative --s 30 --cfg 7.5"
                parts = line.split()
                cleaned_parts = []
                skip_next = False
                
                for i, part in enumerate(parts):
                    if skip_next:
                        skip_next = False
                        continue
                    
                    # Check if this part is a flag
                    if part.startswith('--') or part.startswith('-'):
                        # Check if it's a known flag
                        flag_base = part.split('=')[0]  # Handle --flag=value format
                        if any(flag_base.startswith(f) for f in kohya_flags):
                            # Skip this flag
                            if '=' not in part and i + 1 < len(parts):
                                # Flag has a separate value, skip next part too
                                skip_next = True
                            continue
                    
                    # Not a flag, keep this part
                    cleaned_parts.append(part)
                
                # Reconstruct the prompt without flags
                cleaned_prompt = ' '.join(cleaned_parts).strip()
                
                # Only add non-empty prompts
                if cleaned_prompt:
                    prompts.append(cleaned_prompt)
        
        if not prompts:
            raise ValueError(f"No valid prompts found in file: {file_path}")
        
        print(f"Loaded {len(prompts)} prompts from {file_path.name}")
        return prompts

    def _validate_config(self):
        """Validate configuration structure"""
        if 'xy_plot' not in self.config:
            raise ValueError("Missing 'xy_plot' section in config")
        
        if 'x_axis' not in self.config['xy_plot']:
            raise ValueError("Missing 'x_axis' in xy_plot config")
        
        if 'y_axis' not in self.config['xy_plot']:
            raise ValueError("Missing 'y_axis' in xy_plot config")
    
    def _parse_parameters(self):
        """Parse all parameters to determine which can vary"""
        self.parameters = {}
        
        # Parse checkpoint
        if 'checkpoint' in self.config:
            checkpoint_param = self._parse_checkpoint_parameter()
            if checkpoint_param:
                self.parameters['checkpoint'] = checkpoint_param
        
        # Parse prompts
        if 'prompt' in self.config:
            prompt_param = self._parse_prompt_parameter()
            if prompt_param:
                self.parameters['prompt'] = prompt_param
        
        # Parse LoRAs and LoRA weights
        for key in self.config:
            if key.startswith('lora'):
                # Check if this is specifically a weight variation parameter
                # (has 'weights' plural or ends with '_weight'/'_weights')
                if 'weights' in self.config[key] or key.endswith('_weight') or key.endswith('_weights'):
                    # LoRA weight variations
                    lora_param = self._parse_lora_weight_parameter(key)
                else:
                    # LoRA model variations or single LoRA
                    lora_param = self._parse_lora_parameter(key)
                
                if lora_param:
                    self.parameters[key] = lora_param
        
        # Parse generation parameters that can vary
        for param_name in ['resolution', 'steps', 'cfg_scale', 'seed']:
            if param_name in self.config:
                param = self._parse_generic_parameter(param_name)
                if param:
                    self.parameters[param_name] = param
    
    def _parse_checkpoint_parameter(self) -> Optional[PlotParameter]:
        """Parse checkpoint - can be single (base) or multiple (axis)"""
        checkpoint_config = self.config['checkpoint']
        
        if 'path' in checkpoint_config:
            path = Path(checkpoint_config['path'])
            
            if path.is_dir():
                # Directory = multiple checkpoints
                files = sorted(path.glob("*.safetensors"))
                if files:
                    values = [str(f) for f in files]
                    display_names = [f.stem for f in files]
                    return PlotParameter('checkpoint', values, display_names)
            elif path.is_file():
                # Single file = base model
                return PlotParameter('checkpoint', [str(path)], [path.stem])
        
        elif 'values' in checkpoint_config:
            # Explicit list
            values = checkpoint_config['values']
            display_names = [Path(v).stem for v in values]
            return PlotParameter('checkpoint', values, display_names)
        
        return None
    
    def _parse_prompt_parameter(self) -> Optional[PlotParameter]:
        """Parse prompt parameter - can be from values list or file"""
        prompt_config = self.config['prompt']
        
        # Check for file first (takes priority)
        if 'values_file' in prompt_config:
            file_path = prompt_config['values_file']
            try:
                prompts = self._load_prompts_from_file(file_path)
                # Truncate for display
                display_names = []
                for p in prompts:
                    if len(p) > 50:
                        display_names.append(p[:47] + "...")
                    else:
                        display_names.append(p)
                return PlotParameter('prompt', prompts, display_names)
            except Exception as e:
                print(f"Warning: Failed to load prompts from file: {e}")
                # Fall back to values if file fails
        
        # Use values list if no file or file failed
        if 'values' in prompt_config:
            prompts = prompt_config['values']
            # Handle single prompt as string
            if isinstance(prompts, str):
                prompts = [prompts]
            # Truncate for display
            display_names = [p[:50] + "..." if len(p) > 50 else p for p in prompts]
            return PlotParameter('prompt', prompts, display_names)
        
        # Check if prompt is a single string (not a list)
        if isinstance(prompt_config, str):
            prompts = [prompt_config]
            display_names = [prompt_config[:50] + "..." if len(prompt_config) > 50 else prompt_config]
            return PlotParameter('prompt', prompts, display_names)
        
        return None
    
    def _parse_lora_parameter(self, key: str) -> Optional[PlotParameter]:
        """Parse LoRA model variations"""
        lora_config = self.config[key]
        
        if 'path' in lora_config:
            path = Path(lora_config['path'])
            
            if path.is_dir():
                # Directory of LoRAs
                files = sorted(path.glob("*.safetensors"))
                if files:
                    values = [str(f) for f in files]
                    display_names = [f.stem for f in files]
                    # Store default weight with each LoRA
                    weight = lora_config.get('weight', 1.0)
                    values_with_weight = [(v, weight) for v in values]
                    return PlotParameter(key, values_with_weight, display_names)
            elif path.is_file() or path.suffix == '.safetensors':
                # Single LoRA file (or path that will be a file)
                weight = lora_config.get('weight', 1.0)
                values = [(str(path), weight)]
                display_names = [path.stem]
                return PlotParameter(key, values, display_names)
            else:
                # Path doesn't exist yet, but might be valid
                # Assume it's a single file
                weight = lora_config.get('weight', 1.0)
                values = [(str(path), weight)]
                display_names = [path.stem if path.suffix else path.name]
                return PlotParameter(key, values, display_names)
        
        elif 'values' in lora_config:
            # List of LoRA paths
            values = lora_config['values']
            weight = lora_config.get('weight', 1.0)
            values_with_weight = [(v, weight) for v in values]
            display_names = [Path(v).stem for v in values]
            return PlotParameter(key, values_with_weight, display_names)
        
        return None
    
    def _parse_lora_weight_parameter(self, key: str) -> Optional[PlotParameter]:
        """Parse LoRA weight variations"""
        lora_config = self.config[key]
        
        # Handle both 'weight' (single) and 'weights' (multiple)
        if 'path' in lora_config:
            path = lora_config['path']
            
            # Check for multiple weights
            if 'weights' in lora_config:
                weights = lora_config['weights']
                # Ensure weights is a list
                if not isinstance(weights, list):
                    weights = [weights]
                values = [(path, w) for w in weights]
                display_names = [f"{Path(path).stem} @ {w}" for w in weights]
                return PlotParameter(key, values, display_names)
            
            # Check for single weight (not a variation)
            elif 'weight' in lora_config:
                weight = lora_config['weight']
                values = [(path, weight)]
                display_names = [f"{Path(path).stem} @ {weight}"]
                return PlotParameter(key, values, display_names)
        
        elif 'values' in lora_config:
            # For weight variations specified as tuples
            values = lora_config['values']
            if all(isinstance(v, (list, tuple)) and len(v) == 2 for v in values):
                # Values are (path, weight) tuples
                display_names = [f"{Path(p).stem} @ {w}" for p, w in values]
                return PlotParameter(key, values, display_names)
        
        return None
    
    def _parse_generic_parameter(self, param_name: str) -> Optional[PlotParameter]:
        """Parse generic parameters like resolution, steps, etc."""
        param_config = self.config[param_name]
        
        if 'values' in param_config:
            values = param_config['values']
            return PlotParameter(param_name, values)
        elif isinstance(param_config, list):
            return PlotParameter(param_name, param_config)
        elif isinstance(param_config, (int, float, str)):
            return PlotParameter(param_name, [param_config])
        
        return None
    
    def _parse_axes(self):
        """Parse X and Y axis parameters"""
        x_param_name = self.config['xy_plot']['x_axis']
        y_param_name = self.config['xy_plot']['y_axis']
        
        # Check if parameters exist and are variable
        if x_param_name in self.parameters:
            if self.parameters[x_param_name].is_variable:
                self.x_param = self.parameters[x_param_name]
            else:
                raise ValueError(f"X-axis parameter '{x_param_name}' has only one value")
        else:
            raise ValueError(f"X-axis parameter '{x_param_name}' not found in config")
        
        if y_param_name in self.parameters:
            if self.parameters[y_param_name].is_variable:
                self.y_param = self.parameters[y_param_name]
            else:
                raise ValueError(f"Y-axis parameter '{y_param_name}' has only one value")
        else:
            raise ValueError(f"Y-axis parameter '{y_param_name}' not found in config")
        
        # Validate we don't have too many variables
        variable_params = [p for p in self.parameters.values() if p.is_variable]
        if len(variable_params) > 2:
            param_names = [p.name for p in variable_params]
            raise ValueError(f"Too many variable parameters: {param_names}. Maximum 2 allowed.")
        
        print(f"X-axis: {self.x_param.name} with {len(self.x_param.values)} values")
        print(f"Y-axis: {self.y_param.name} with {len(self.y_param.values)} values")
    
    def get_base_config(self) -> Dict[str, Any]:
        """Get base configuration for all generations"""
        base_config = {}
        
        # Add generation parameters
        if 'generation' in self.config:
            base_config.update(self.config['generation'])
        
        # Add non-variable parameters
        for param_name, param in self.parameters.items():
            if not param.is_variable:
                if param_name == 'checkpoint':
                    base_config['checkpoint'] = param.values[0]
                elif param_name == 'prompt':
                    base_config['prompt'] = param.values[0]
                elif param_name.startswith('lora'):
                    # Single LoRA
                    if isinstance(param.values[0], tuple):
                        path, weight = param.values[0]
                        base_config[f'{param_name}_path'] = path
                        base_config[f'{param_name}_weight'] = weight
                else:
                    base_config[param_name] = param.values[0]
        
        return base_config
    
    def create_generation_configs(self) -> List[Tuple[Dict[str, Any], str, str]]:
        """Create all generation configurations for the grid"""
        configs = []
        base_config = self.get_base_config()
        
        # Create cartesian product of X and Y values
        for y_idx, y_value in enumerate(self.y_param.values):
            for x_idx, x_value in enumerate(self.x_param.values):
                # Start with base config
                gen_config = base_config.copy()
                
                # Apply X parameter
                self._apply_parameter(gen_config, self.x_param.name, x_value)
                
                # Apply Y parameter
                self._apply_parameter(gen_config, self.y_param.name, y_value)
                
                # Get display names
                x_label = self.x_param.display_names[x_idx]
                y_label = self.y_param.display_names[y_idx]
                
                configs.append((gen_config, x_label, y_label))
        
        return configs
    
    def _apply_parameter(self, config: Dict[str, Any], param_name: str, value: Any):
        """Apply a parameter value to the generation config"""
        if param_name == 'prompt':
            config['prompt'] = value
        
        elif param_name == 'checkpoint':
            config['checkpoint'] = value
        
        elif param_name.startswith('lora'):
            # LoRA with or without weight
            if isinstance(value, tuple):
                path, weight = value
                config[f'{param_name}_path'] = path
                config[f'{param_name}_weight'] = weight
            else:
                config[f'{param_name}_path'] = value
                config[f'{param_name}_weight'] = 1.0
        
        elif param_name in ['resolution', 'steps', 'cfg_scale', 'seed']:
            config[param_name] = value


class FlexibleXYPlotGenerator:
    """Generate XY plots with flexible parameter configuration"""
    
    def __init__(self, config_path: str):
        self.plot_config = XYPlotConfig(config_path)
        self.config = self.plot_config.config
        self.output_dir = Path(self.config.get('output', {}).get('output_dir', './output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize generator
        self._init_generator()
    
    def _init_generator(self):
        """Initialize the SDXL generator"""
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
            enable_cpu_offload = gpu_memory_gb < 12
        else:
            enable_cpu_offload = False
        
        memory_config = self.config.get('memory_optimization', {})
        enable_cpu_offload = memory_config.get('enable_cpu_offload', enable_cpu_offload)
        
        self.generator = SDXLGenerator(
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            enable_cpu_offload=enable_cpu_offload,
            enable_attention_slicing=memory_config.get('enable_attention_slicing', True),
            enable_vae_tiling=memory_config.get('enable_vae_tiling', True)
        )
    
    def generate_grid(self):
        """Generate the XY plot grid"""
        # Get all generation configurations
        gen_configs = self.plot_config.create_generation_configs()
        
        # Get dimensions
        x_values = self.plot_config.x_param.display_names
        y_values = self.plot_config.y_param.display_names
        
        print(f"\nGenerating {len(x_values)}x{len(y_values)} grid = {len(gen_configs)} images")
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create individual images directory if needed
        individual_dir = None
        if self.config.get('output', {}).get('save_individual_images', False):
            individual_dir = self.output_dir / f"individual_{timestamp}"
            individual_dir.mkdir(parents=True, exist_ok=True)
            print(f"Individual images will be saved to: {individual_dir}")
        
        # Set scheduler
        gen_config = self.config.get('generation', {})
        scheduler = gen_config.get('scheduler', 'EulerDiscreteScheduler')
        self.generator.set_scheduler(scheduler)
        
        # Generate images
        grid_images = []
        
        with tqdm(total=len(gen_configs), desc="Generating images") as pbar:
            for y_idx, y_label in enumerate(y_values):
                row_images = []
                
                for x_idx, x_label in enumerate(x_values):
                    config_idx = y_idx * len(x_values) + x_idx
                    gen_config, _, _ = gen_configs[config_idx]
                    
                    pbar.set_description(f"Y: {y_label[:20]} | X: {x_label[:20]}")
                    
                    # Generate image based on config
                    image = self._generate_from_config(gen_config)
                    
                    if image is None:
                        # Create placeholder
                        resolution = gen_config.get('resolution', 1024)
                        image = Image.new('RGB', (resolution, resolution), color='lightgray')
                    
                    # Save individual image if requested
                    if individual_dir:
                        img_name = f"y{y_idx:02d}_x{x_idx:02d}_{y_label[:20]}_{x_label[:20]}.png"
                        img_name = img_name.replace('/', '_').replace('\\', '_')  # Clean filename
                        img_path = individual_dir / img_name
                        image.save(img_path, quality=95)
                        
                        # Save metadata
                        meta_path = individual_dir / f"y{y_idx:02d}_x{x_idx:02d}_metadata.json"
                        with open(meta_path, 'w') as f:
                            json.dump({
                                'x_label': x_label,
                                'y_label': y_label,
                                'config': gen_config
                            }, f, indent=2)
                    
                    row_images.append(image)
                    pbar.update(1)
                
                grid_images.append(row_images)
        
        # Create grid image
        print("\nCreating grid image...")
        grid_img = self._create_grid_image(
            grid_images,
            x_values,
            y_values,
            self.plot_config.config['xy_plot'].get('x_axis_label', self.plot_config.x_param.name),
            self.plot_config.config['xy_plot'].get('y_axis_label', self.plot_config.y_param.name)
        )
        
        # Save outputs
        grid_path = self.output_dir / f"xy_plot_{timestamp}.png"
        grid_img.save(grid_path, quality=95)
        print(f"Grid saved to: {grid_path}")
        
        # Create preview if requested
        if self.config.get('output', {}).get('create_preview', True):
            preview_scale = self.config.get('output', {}).get('preview_scale', 0.25)
            preview_size = (int(grid_img.width * preview_scale), int(grid_img.height * preview_scale))
            preview = grid_img.resize(preview_size, Image.Resampling.LANCZOS)
            preview_path = self.output_dir / f"xy_plot_{timestamp}_preview.png"
            preview.save(preview_path)
            print(f"Preview saved to: {preview_path}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'x_axis': self.plot_config.x_param.name,
            'y_axis': self.plot_config.y_param.name,
            'x_values': x_values,
            'y_values': y_values,
            'config': self.config
        }
        metadata_path = self.output_dir / f"xy_plot_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
    
    def _generate_from_config(self, gen_config: Dict[str, Any]) -> Optional[Image.Image]:
        """Generate image from a configuration"""
        # Determine what needs to be loaded
        has_checkpoint = 'checkpoint' in gen_config
        has_lora1 = 'lora1_path' in gen_config
        has_lora2 = 'lora2_path' in gen_config
        
        # Load checkpoint
        if has_checkpoint:
            success = self.generator.load_checkpoint(gen_config['checkpoint'])
            if not success:
                print(f"Failed to load checkpoint: {gen_config['checkpoint']}")
                return None
        else:
            # No checkpoint specified - error
            print("Error: No checkpoint specified")
            return None
        
        # Apply LoRAs if present
        if has_lora1:
            lora1_path = gen_config['lora1_path']
            lora1_weight = gen_config.get('lora1_weight', 1.0)
            success = self.generator.apply_lora_to_current_model(
                lora1_path, lora1_weight, adapter_name="lora1"
            )
            if not success:
                print(f"Failed to apply LoRA1: {lora1_path}")
        
        if has_lora2:
            lora2_path = gen_config['lora2_path']
            lora2_weight = gen_config.get('lora2_weight', 1.0)
            success = self.generator.apply_lora_to_current_model(
                lora2_path, lora2_weight, adapter_name="lora2"
            )
            if not success:
                print(f"Failed to apply LoRA2: {lora2_path}")
        
        # Generate with parameters
        return self.generator.generate(
            prompt=gen_config.get('prompt', 'a photo'),
            negative_prompt=gen_config.get('negative_prompt', ''),
            width=gen_config.get('resolution', 1024),
            height=gen_config.get('resolution', 1024),
            num_inference_steps=gen_config.get('steps', 30),
            guidance_scale=gen_config.get('cfg_scale', 7.0),
            seed=gen_config.get('seed', 42),
            clip_skip=gen_config.get('clip_skip', 1)
        )
    
    def _create_grid_image(
        self,
        images: List[List[Image.Image]],
        x_labels: List[str],
        y_labels: List[str],
        x_axis_title: str,
        y_axis_title: str
    ) -> Image.Image:
        """Create grid image with proper text wrapping for both axes"""
        
        grid_config = self.config.get('grid', {})
        gen_config = self.config.get('generation', {})
        resolution = gen_config.get('resolution', 1024)
        margin = grid_config.get('margin_size', 250)
        
        num_x = len(x_labels)
        num_y = len(y_labels)
        
        # Calculate canvas size
        canvas_width = margin + (num_x * resolution)
        canvas_height = margin + (num_y * resolution)
        
        # Create canvas
        bg_color = grid_config.get('background_color', '#ffffff')
        canvas = Image.new('RGB', (canvas_width, canvas_height), bg_color)
        draw = ImageDraw.Draw(canvas)
        
        # Load font
        font_size = grid_config.get('label_font_size', 20)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
            title_font = ImageFont.truetype("arial.ttf", font_size + 4)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size + 4)
            except:
                font = ImageFont.load_default()
                title_font = font
        
        # Configuration
        font_color = grid_config.get('label_font_color', '#000000')
        max_chars = grid_config.get('max_label_chars', 200)
        line_spacing = grid_config.get('label_line_spacing', 4)
        line_color = grid_config.get('grid_line_color', '#cccccc')
        line_width = grid_config.get('grid_line_width', 2)
        
        def wrap_text(text, max_width):
            """Wrap text to fit within max_width pixels"""
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            words = text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=font)
                
                if bbox[2] - bbox[0] <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            return lines if lines else [text]
        
        def draw_wrapped_text(x, y, text, max_width, align='left'):
            """Draw wrapped text"""
            lines = wrap_text(text, max_width)
            
            if align == 'center':
                # For Y-axis labels (vertical centering)
                total_height = len(lines) * (font_size + line_spacing)
                current_y = y - total_height // 2
            else:
                # For X-axis labels (top alignment)
                current_y = y
            
            for line in lines:
                draw.text((x, current_y), line, fill=font_color, font=font)
                current_y += font_size + line_spacing
        
        # Draw axis titles
        draw.text((margin // 2, 10), y_axis_title, fill=font_color, font=title_font, anchor="mt")
        draw.text((canvas_width // 2, 10), x_axis_title, fill=font_color, font=title_font, anchor="mt")
        
        # Draw X-axis labels (top)
        for i, label in enumerate(x_labels):
            x = margin + (i * resolution)
            # Draw background rectangle for label
            draw.rectangle([(x, 35), (x + resolution, margin - 5)], fill=bg_color)
            draw_wrapped_text(x + 10, 40, label, resolution - 20, align='left')
            
            if i > 0:
                draw.line([(x, margin), (x, canvas_height)], fill=line_color, width=line_width)
        
        # Draw Y-axis labels (left)
        for i, label in enumerate(y_labels):
            y = margin + (i * resolution)
            # Draw background rectangle for label
            draw.rectangle([(0, y), (margin - 5, y + resolution)], fill=bg_color)
            draw_wrapped_text(10, y + resolution // 2, label, margin - 20, align='center')
            
            if i > 0:
                draw.line([(margin, y), (canvas_width, y)], fill=line_color, width=line_width)
        
        # Draw separator lines
        draw.line([(margin - 5, 35), (margin - 5, canvas_height)], fill=line_color, width=line_width)
        draw.line([(0, margin - 5), (canvas_width, margin - 5)], fill=line_color, width=line_width)
        
        # Paste images
        for y_idx, row_images in enumerate(images):
            for x_idx, image in enumerate(row_images):
                x = margin + (x_idx * resolution)
                y = margin + (y_idx * resolution)
                
                if image.size != (resolution, resolution):
                    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
                
                canvas.paste(image, (x, y))
        
        return canvas