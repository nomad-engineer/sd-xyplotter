"""
Lightweight SDXL image generator using Diffusers with proper LoRA support
"""

import os
import gc
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image
import json
from datetime import datetime
from tqdm.auto import tqdm
import sys

from diffusers import (
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    EulerDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from safetensors.torch import load_file

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class SDXLGenerator:
    """Lightweight SDXL generator using Diffusers library with proper LoRA support"""
    
    SCHEDULERS = {
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "DDIMScheduler": DDIMScheduler,
        "PNDMScheduler": PNDMScheduler,
        "LMSDiscreteScheduler": LMSDiscreteScheduler,
        "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
        "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    }
    
    def __init__(self, device: str = None, dtype: torch.dtype = torch.float16, 
                 enable_cpu_offload: bool = True, enable_attention_slicing: bool = True,
                 enable_vae_tiling: bool = True, lora_strength: float = 1.0, 
                 clip_strength: float = 1.0):
        """Initialize the generator with memory optimization options"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.pipe = None
        self.current_model_path = None
        self.base_model_path = None
        self.is_lora_mode = False
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_attention_slicing = enable_attention_slicing
        self.enable_vae_tiling = enable_vae_tiling
        self.current_lora_path = None
        self.lora_strength = lora_strength
        self.clip_strength = clip_strength
        self.lora_adapter_names = []  # Track loaded adapter names
        
        # Clear any existing cache
        self._clear_memory()
        
    def _clear_memory(self):
        """Clear GPU memory cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def load_model(self, model_path: str, is_lora: bool = False, base_model_path: str = None):
        """Load a model (checkpoint or LoRA) with proper LoRA support"""
        model_path = Path(model_path)
        
        try:
            if is_lora:
                # For LoRA mode, we need a base model
                if base_model_path is None:
                    raise ValueError("Base model path required for LoRA mode")
                
                # Load base model if not already loaded or different
                if self.pipe is None or self.base_model_path != base_model_path:
                    print(f"Loading base model: {Path(base_model_path).name}")
                    
                    # Clear any existing pipe
                    if self.pipe is not None:
                        # Unload any previous LoRA
                        self._unload_all_loras()
                        del self.pipe
                        self._clear_memory()
                    
                    self.pipe = self._load_checkpoint(base_model_path)
                    self.base_model_path = base_model_path
                    self.current_lora_path = None
                    self.lora_adapter_names = []
                
                # Check if this is a different LoRA than currently loaded
                if self.current_lora_path != str(model_path):
                    # Unload all previous LoRAs
                    if self.lora_adapter_names:
                        print(f"Unloading previous LoRAs: {self.lora_adapter_names}")
                        self._unload_all_loras()
                    
                    # Load new LoRA
                    print(f"Loading LoRA: {model_path.name} with strength={self.lora_strength}")
                    adapter_name = self._load_lora(str(model_path))
                    self.current_lora_path = str(model_path)
                    
                    # Verify LoRA was applied
                    print(f"LoRA {model_path.name} loaded with adapter name: {adapter_name}")
                else:
                    print(f"LoRA {model_path.name} already loaded")
                
            else:
                # Load full checkpoint
                if self.current_model_path != str(model_path):
                    print(f"Loading checkpoint: {model_path.name}")
                    
                    # Clear previous models
                    if self.pipe is not None:
                        self._unload_all_loras()
                        del self.pipe
                        self._clear_memory()
                    
                    self.pipe = self._load_checkpoint(str(model_path))
                    self.current_lora_path = None
                    self.lora_adapter_names = []
                    print(f"Successfully loaded checkpoint: {model_path.name}")
                else:
                    print(f"Checkpoint {model_path.name} already loaded")
                
            self.current_model_path = str(model_path)
            self.is_lora_mode = is_lora
            return True  # Success
            
        except Exception as e:
            print(f"Failed to load model {model_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return False  # Failure
    
    def _load_lora(self, lora_path: str) -> str:
        """Load LoRA weights and return the adapter name"""
        if self.pipe is None:
            raise ValueError("No base model loaded")
        
        try:
            # Generate a unique adapter name based on the file
            adapter_name = Path(lora_path).stem
            
            # Load LoRA weights with a specific adapter name
            self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            
            # Track the adapter name
            self.lora_adapter_names.append(adapter_name)
            
            # Set the adapter with specified strength
            # For SDXL, we might have separate strengths for UNet and text encoders
            if hasattr(self.pipe, 'set_adapters'):
                try:
                    # Try setting with the specific adapter name
                    self.pipe.set_adapters([adapter_name], adapter_weights=[self.lora_strength])
                except ValueError as e:
                    # If that fails, check what adapters are actually present
                    print(f"Checking available adapters...")
                    
                    # Get all adapter names
                    all_adapters = set()
                    if hasattr(self.pipe.unet, 'peft_config'):
                        all_adapters.update(self.pipe.unet.peft_config.keys())
                    if hasattr(self.pipe, 'text_encoder') and hasattr(self.pipe.text_encoder, 'peft_config'):
                        all_adapters.update(self.pipe.text_encoder.peft_config.keys())
                    if hasattr(self.pipe, 'text_encoder_2') and hasattr(self.pipe.text_encoder_2, 'peft_config'):
                        all_adapters.update(self.pipe.text_encoder_2.peft_config.keys())
                    
                    print(f"Available adapters: {all_adapters}")
                    
                    # Use all available adapters
                    if all_adapters:
                        adapter_list = list(all_adapters)
                        weights = [self.lora_strength] * len(adapter_list)
                        self.pipe.set_adapters(adapter_list, adapter_weights=weights)
                        print(f"Set adapters: {adapter_list} with weights: {weights}")
                        self.lora_adapter_names = adapter_list
                    else:
                        print("Warning: No adapters found, LoRA may not be applied correctly")
            
            # Alternative: Use scale if set_adapters doesn't work
            if hasattr(self.pipe, '_lora_scale'):
                self.pipe._lora_scale = self.lora_strength
                print(f"Set LoRA scale to {self.lora_strength}")
            
            return adapter_name
            
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            
            # Fallback: Try fusing LoRA if loading fails
            try:
                print("Attempting to fuse LoRA as fallback...")
                self.pipe.fuse_lora(lora_scale=self.lora_strength)
                return "fused"
            except:
                raise e
    
    def _unload_all_loras(self):
        """Unload all current LoRA weights"""
        if self.pipe is None or not self.lora_adapter_names:
            return
        
        try:
            # Try to unload using diffusers method
            if hasattr(self.pipe, 'unload_lora_weights'):
                self.pipe.unload_lora_weights()
                print("Unloaded all LoRA weights")
            
            # Also try to disable adapters
            if hasattr(self.pipe, 'disable_adapters'):
                self.pipe.disable_adapters()
                
        except Exception as e:
            print(f"Error unloading LoRAs: {e}")
            # If unloading fails, we'll need to reload the base model on next LoRA load
        
        self.lora_adapter_names = []
        self.current_lora_path = None
    
    def _load_checkpoint(self, checkpoint_path: str) -> StableDiffusionXLPipeline:
        """Load a full SDXL checkpoint with memory optimizations"""
        print("Loading with memory optimizations enabled...")
        
        # Load with lower precision and CPU first to save memory
        load_kwargs = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
            "variant": "fp16" if self.dtype == torch.float16 else None,
            "low_cpu_mem_usage": True,
        }
        
        # Check if it's a single safetensors file or a diffusers directory
        if checkpoint_path.endswith('.safetensors'):
            # Load from single file
            pipe = StableDiffusionXLPipeline.from_single_file(
                checkpoint_path,
                **load_kwargs
            )
        else:
            # Load from diffusers format directory
            pipe = StableDiffusionXLPipeline.from_pretrained(
                checkpoint_path,
                **load_kwargs
            )
        
        # Apply memory optimizations
        if self.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
            print("Enabled CPU offload")
        else:
            pipe = pipe.to(self.device)
        
        if self.enable_attention_slicing:
            pipe.enable_attention_slicing(slice_size="auto")
            print("Enabled attention slicing")
        
        if self.enable_vae_tiling:
            pipe.enable_vae_tiling()
            print("Enabled VAE tiling")
        
        # Enable xformers if available
        if hasattr(pipe.unet, 'enable_xformers_memory_efficient_attention'):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("Enabled xformers memory efficient attention")
            except Exception as e:
                print(f"Could not enable xformers: {e}")
        
        return pipe
        
    def set_scheduler(self, scheduler_name: str):
        """Set the scheduler for the pipeline"""
        if self.pipe is None:
            return
            
        if scheduler_name in self.SCHEDULERS:
            scheduler_class = self.SCHEDULERS[scheduler_name]
            self.pipe.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)
            print(f"Set scheduler to {scheduler_name}")
            
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        seed: int = -1,
        clip_skip: int = 1
    ) -> Optional[Image.Image]:
        """Generate a single image with memory management"""
        if self.pipe is None:
            print("No model loaded for generation")
            return None
        
        try:
            # Set seed
            generator = None
            if seed >= 0:
                generator = torch.Generator(device="cpu").manual_seed(seed)
            
            # Set clip skip
            if clip_skip > 1 and hasattr(self.pipe, 'text_encoder'):
                self.pipe.text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
            
            # Debug: Print current model info
            if self.is_lora_mode and self.current_lora_path:
                print(f"Generating with LoRA: {Path(self.current_lora_path).name} (strength={self.lora_strength})")
            
            # Set cross attention kwargs for LoRA if needed
            cross_attention_kwargs = {}
            if self.lora_adapter_names and hasattr(self.pipe, '_lora_scale'):
                cross_attention_kwargs["scale"] = self.lora_strength
            
            # Generate image with automatic mixed precision
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        cross_attention_kwargs=cross_attention_kwargs if cross_attention_kwargs else None,
                    )
            
            image = result.images[0]
            
            # Clear intermediate tensors
            del result
            self._clear_memory()
            
            return image
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA OOM during generation. Attempting recovery...")
            self._clear_memory()
            
            # Try with reduced settings
            if width > 768 or height > 768:
                print(f"Retrying with reduced resolution (768x768)...")
                width = height = 768
                
                with torch.cuda.amp.autocast(enabled=True):
                    with torch.no_grad():
                        result = self.pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        )
                
                return result.images[0]
            else:
                print("Failed to generate even with reduced settings")
                return None
                
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return None


class XYPlotGenerator:
    """Generate XY plots from multiple models and prompts with memory management"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        self.config = config
        
        # Check available memory and adjust settings
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
            
            # Auto-enable memory optimizations for GPUs with less than 12GB VRAM
            enable_cpu_offload = gpu_memory_gb < 12
            enable_attention_slicing = gpu_memory_gb < 16
            enable_vae_tiling = gpu_memory_gb < 12
        else:
            enable_cpu_offload = False
            enable_attention_slicing = True
            enable_vae_tiling = True
        
        # Allow config overrides
        memory_config = config.get('memory_optimization', {})
        enable_cpu_offload = memory_config.get('enable_cpu_offload', enable_cpu_offload)
        enable_attention_slicing = memory_config.get('enable_attention_slicing', enable_attention_slicing)
        enable_vae_tiling = memory_config.get('enable_vae_tiling', enable_vae_tiling)
        
        # Get LoRA settings
        lora_config = config.get('lora', {})
        lora_strength = lora_config.get('lora_strength', 1.0)
        clip_strength = lora_config.get('clip_strength', 1.0)
        
        print(f"Memory optimizations: CPU offload={enable_cpu_offload}, "
              f"Attention slicing={enable_attention_slicing}, VAE tiling={enable_vae_tiling}")
        
        if lora_strength != 1.0 or clip_strength != 1.0:
            print(f"LoRA settings: strength={lora_strength}, clip_strength={clip_strength}")
        
        self.generator = SDXLGenerator(
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            enable_cpu_offload=enable_cpu_offload,
            enable_attention_slicing=enable_attention_slicing,
            enable_vae_tiling=enable_vae_tiling,
            lora_strength=lora_strength,
            clip_strength=clip_strength
        )
        self.output_dir = Path(config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_prompts(self) -> List[str]:
        """Load prompts from file"""
        prompts_file = Path(self.config['paths']['prompts_file'])
        if not prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
            
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
            
        print(f"Loaded {len(prompts)} prompts")
        return prompts
        
    def find_models(self) -> List[Tuple[Path, str]]:
        """Find all model files in the model directory"""
        model_dir = Path(self.config['paths']['model_dir'])
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        # Find safetensors files
        model_files = sorted(model_dir.glob("*.safetensors"))
        
        if not model_files:
            # Try looking for diffusers format directories
            model_files = [d for d in model_dir.iterdir() if d.is_dir() and (d / "model_index.json").exists()]
            
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
            
        models = [(f, f.stem if f.is_file() else f.name) for f in model_files]
        print(f"Found {len(models)} models")
        return models
        
    def generate_grid(self):
        """Generate the XY plot grid with memory management"""
        # Load prompts and find models
        prompts = self.load_prompts()
        models = self.find_models()
        
        # Setup configuration
        gen_config = self.config['generation']
        resolution = gen_config['resolution']
        
        # Determine if using LoRAs
        base_checkpoint = self.config['paths'].get('base_checkpoint', '')
        is_lora_mode = bool(base_checkpoint)
        
        if is_lora_mode:
            print(f"LoRA mode enabled with base model: {Path(base_checkpoint).name}")
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create individual images directory if needed
        individual_dir = None
        if self.config['output']['save_individual_images']:
            individual_dir = self.output_dir / f"{self.config['output']['individual_images_dir']}_{timestamp}"
            individual_dir.mkdir(parents=True, exist_ok=True)
            print(f"Individual images will be saved to: {individual_dir}")
        
        # Set scheduler
        self.generator.set_scheduler(gen_config.get('scheduler', 'EulerDiscreteScheduler'))
        
        # Track successful models
        successful_models = []
        grid_images = []
        
        # First pass: try to load all models and track which ones succeed
        print("\nValidating models...")
        for model_path, model_name in models:
            success = self.generator.load_model(
                str(model_path),
                is_lora=is_lora_mode,
                base_model_path=base_checkpoint if is_lora_mode else None
            )
            
            if success:
                successful_models.append((model_path, model_name))
            else:
                print(f"Skipping model {model_name} due to loading error")
        
        if not successful_models:
            print("ERROR: No models could be loaded successfully!")
            sys.exit(1)
        
        print(f"\nSuccessfully validated {len(successful_models)} out of {len(models)} models")
        
        # Generate images only for successful models
        total = len(successful_models) * len(prompts)
        
        with tqdm(total=total, desc="Generating images") as pbar:
            for model_idx, (model_path, model_name) in enumerate(successful_models):
                # Load model (should succeed since we validated)
                self.generator.load_model(
                    str(model_path),
                    is_lora=is_lora_mode,
                    base_model_path=base_checkpoint if is_lora_mode else None
                )
                
                model_images = []
                
                for prompt_idx, prompt in enumerate(prompts):
                    pbar.set_description(f"{model_name[:20]} - Prompt {prompt_idx+1}/{len(prompts)}")
                    
                    # Generate image
                    image = self.generator.generate(
                        prompt=prompt,
                        negative_prompt=gen_config['negative_prompt'],
                        width=resolution,
                        height=resolution,
                        num_inference_steps=gen_config['steps'],
                        guidance_scale=gen_config['cfg_scale'],
                        seed=gen_config['seed'],
                        clip_skip=gen_config.get('clip_skip', 1)
                    )
                    
                    if image is None:
                        # Generation failed, create placeholder
                        print(f"\nFailed to generate image for {model_name} with prompt {prompt_idx}")
                        image = Image.new('RGB', (resolution, resolution), color='lightgray')
                    else:
                        # Save individual image if requested
                        if individual_dir:
                            img_name = f"{model_name}_{prompt_idx:03d}.png"
                            img_path = individual_dir / img_name
                            image.save(img_path, quality=self.config['output']['grid_image_quality'])
                            
                            # Save prompt info with LoRA strength
                            info_path = individual_dir / f"{model_name}_{prompt_idx:03d}.txt"
                            lora_info = self.config.get('lora', {})
                            with open(info_path, 'w', encoding='utf-8') as f:
                                f.write(f"model,prompt_idx,prompt,negative_prompt,seed,steps,cfg_scale,lora_strength,clip_strength\n")
                                f.write(f'"{model_name}",{prompt_idx},"{prompt}","{gen_config["negative_prompt"]}",{gen_config["seed"]},{gen_config["steps"]},{gen_config["cfg_scale"]},{lora_info.get("lora_strength", 1.0)},{lora_info.get("clip_strength", 1.0)}\n')
                    
                    model_images.append(image)
                    pbar.update(1)
                
                grid_images.append(model_images)
                
                # Clear memory after each model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Create the grid only with successful models
        print("\nCreating grid image...")
        grid_img = self._create_grid_image(grid_images, successful_models, prompts)
        
        # Save grid
        grid_path = self.output_dir / f"xy_grid_{timestamp}.png"
        grid_img.save(grid_path, quality=self.config['output']['grid_image_quality'])
        print(f"Grid saved to: {grid_path}")
        
        # Create preview if requested
        if self.config['output']['create_preview']:
            preview_scale = self.config['output']['preview_scale']
            preview_size = (int(grid_img.width * preview_scale), int(grid_img.height * preview_scale))
            preview = grid_img.resize(preview_size, Image.Resampling.LANCZOS)
            preview_path = self.output_dir / f"xy_grid_{timestamp}_preview.png"
            preview.save(preview_path)
            print(f"Preview saved to: {preview_path}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'models_attempted': [m[1] for m in models],
            'models_successful': [m[1] for m in successful_models],
            'prompts': prompts,
            'config': self.config,
            'grid_size': {'width': grid_img.width, 'height': grid_img.height}
        }
        metadata_path = self.output_dir / f"xy_grid_{timestamp}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
        
def _create_grid_image(
    self,
    images: List[List[Image.Image]],
    models: List[Tuple[Path, str]],
    prompts: List[str]
) -> Image.Image:
    """Create the grid image with labels"""
    from PIL import ImageDraw, ImageFont
    import textwrap
    
    resolution = self.config['generation']['resolution']
    margin = self.config['grid']['margin_size']
    
    num_models = len(models)
    num_prompts = len(prompts)
    
    # Calculate canvas size
    canvas_width = margin + (num_prompts * resolution)
    canvas_height = margin + (num_models * resolution)
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load font
    font_size = self.config['grid']['label_font_size']
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            # Try alternative font locations
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Get configuration
    line_color = self.config['grid']['grid_line_color']
    line_width = self.config['grid']['grid_line_width']
    font_color = self.config['grid'].get('label_font_color', '#000000')
    max_chars = self.config['grid'].get('max_label_chars', 150)
    line_spacing = self.config['grid'].get('label_line_spacing', 5)
    
    def wrap_text(text, max_width, font):
        """Wrap text to fit within max_width pixels"""
        # First check if we need to truncate
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        # Try to fit text as-is
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return [text]
        
        # Need to wrap
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
                else:
                    # Word is too long, need to break it
                    for i in range(len(word)):
                        if i > 0:
                            test_word = word[:i] + "..."
                            bbox = draw.textbbox((0, 0), test_word, font=font)
                            if bbox[2] - bbox[0] > max_width:
                                lines.append(word[:i-1] + "...")
                                break
                    else:
                        lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text[:20] + "..."]
    
    def draw_wrapped_text(x, y, text, max_width, font, fill, anchor="lt"):
        """Draw wrapped text at position"""
        lines = wrap_text(text, max_width, font)
        
        if anchor == "lt":  # left-top
            current_y = y
            for line in lines:
                draw.text((x, current_y), line, fill=fill, font=font)
                bbox = draw.textbbox((0, 0), line, font=font)
                current_y += (bbox[3] - bbox[1]) + line_spacing
        elif anchor == "lm":  # left-middle
            # Calculate total height
            total_height = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                total_height += (bbox[3] - bbox[1]) + line_spacing
            total_height -= line_spacing  # Remove last spacing
            
            # Draw centered vertically
            current_y = y - total_height // 2
            for line in lines:
                draw.text((x, current_y), line, fill=fill, font=font)
                bbox = draw.textbbox((0, 0), line, font=font)
                current_y += (bbox[3] - bbox[1]) + line_spacing
    
    # Draw prompt headers (horizontal labels at top)
    for i, prompt in enumerate(prompts):
        x = margin + (i * resolution)
        # Draw background for better readability
        draw.rectangle([(x, 0), (x + resolution, margin - 5)], fill='white')
        
        # Draw wrapped text
        draw_wrapped_text(
            x + 10, 
            10, 
            prompt, 
            resolution - 20,  # Leave some padding
            font, 
            font_color, 
            anchor="lt"
        )
        
        # Draw vertical separator line
        if i > 0:
            draw.line([(x, margin), (x, canvas_height)], fill=line_color, width=line_width)
    
    # Draw horizontal line under headers
    draw.line([(0, margin - 5), (canvas_width, margin - 5)], fill=line_color, width=line_width)
    
    # Draw model labels (vertical labels on left)
    for i, (_, model_name) in enumerate(models):
        y = margin + (i * resolution)
        
        # Draw background for better readability
        draw.rectangle([(0, y), (margin - 5, y + resolution)], fill='white')
        
        # Draw wrapped text (vertically centered)
        draw_wrapped_text(
            10, 
            y + resolution // 2, 
            model_name, 
            margin - 20,  # Leave some padding
            font, 
            font_color, 
            anchor="lm"
        )
        
        # Draw horizontal separator line
        if i > 0:
            draw.line([(margin, y), (canvas_width, y)], fill=line_color, width=line_width)
    
    # Draw vertical line after labels
    draw.line([(margin - 5, 0), (margin - 5, canvas_height)], fill=line_color, width=line_width)
    
    # Paste images
    for model_idx, model_images in enumerate(images):
        for prompt_idx, image in enumerate(model_images):
            x = margin + (prompt_idx * resolution)
            y = margin + (model_idx * resolution)
            
            if image.size != (resolution, resolution):
                image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
                
            canvas.paste(image, (x, y))
    
    return canvas