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
                 enable_vae_tiling: bool = True):
        """Initialize the generator with memory optimization options"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.pipe = None
        self.current_model_path = None
        self.is_lora_mode = False
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_attention_slicing = enable_attention_slicing
        self.enable_vae_tiling = enable_vae_tiling
        self.current_lora_paths = {}  # Track loaded LoRAs
        self.lora_adapter_names = []
        
        # Clear any existing cache
        self._clear_memory()
        
    def _clear_memory(self):
        """Clear GPU memory cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint model"""
        checkpoint_path = Path(checkpoint_path)
        
        # Skip if already loaded
        if self.current_model_path == str(checkpoint_path):
            return True
            
        try:
            print(f"Loading checkpoint: {checkpoint_path.name}")
            
            # Clear previous model
            if self.pipe is not None:
                self._unload_all_loras()
                del self.pipe
                self._clear_memory()
            
            self.pipe = self._load_checkpoint(str(checkpoint_path))
            self.current_model_path = str(checkpoint_path)
            self.current_lora_paths = {}
            self.lora_adapter_names = []
            
            print(f"Successfully loaded checkpoint: {checkpoint_path.name}")
            return True
            
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path.name}: {e}")
            return False
    
    def apply_lora_to_current_model(self, lora_path: str, weight: float = 1.0, adapter_name: str = None):
        """Apply a LoRA to the currently loaded checkpoint"""
        if self.pipe is None:
            raise ValueError("No checkpoint loaded")
        
        lora_path = Path(lora_path)
        if adapter_name is None:
            adapter_name = lora_path.stem
        
        # Check if this LoRA is already loaded with same weight
        if adapter_name in self.current_lora_paths:
            if self.current_lora_paths[adapter_name] == (str(lora_path), weight):
                return True
        
        try:
            print(f"Applying LoRA {lora_path.name} (weight={weight}) to {Path(self.current_model_path).name}")
            
            # Unload this specific LoRA if already loaded with different settings
            if adapter_name in self.lora_adapter_names:
                # For now, we need to unload all and reapply
                # This is a limitation of current diffusers
                self._unload_all_loras()
            
            # Load LoRA weights
            self.pipe.load_lora_weights(str(lora_path), adapter_name=adapter_name)
            
            # Track this LoRA
            self.current_lora_paths[adapter_name] = (str(lora_path), weight)
            if adapter_name not in self.lora_adapter_names:
                self.lora_adapter_names.append(adapter_name)
            
            # Set adapter weights for all loaded LoRAs
            if self.lora_adapter_names:
                weights = []
                for name in self.lora_adapter_names:
                    _, w = self.current_lora_paths[name]
                    weights.append(w)
                
                try:
                    self.pipe.set_adapters(self.lora_adapter_names, adapter_weights=weights)
                except ValueError:
                    # Handle the case where adapter names might be different
                    # Try to get actual adapter names from the pipe
                    all_adapters = set()
                    if hasattr(self.pipe.unet, 'peft_config'):
                        all_adapters.update(self.pipe.unet.peft_config.keys())
                    
                    if all_adapters:
                        # Use whatever adapters are actually loaded
                        adapter_list = list(all_adapters)
                        weights = [weight] * len(adapter_list)
                        self.pipe.set_adapters(adapter_list, adapter_weights=weights)
            
            return True
            
        except Exception as e:
            print(f"Failed to apply LoRA {lora_path.name}: {e}")
            return False
    
    def _unload_all_loras(self):
        """Unload all current LoRA weights"""
        if self.pipe is None or not self.lora_adapter_names:
            return
        
        try:
            if hasattr(self.pipe, 'unload_lora_weights'):
                self.pipe.unload_lora_weights()
                print("Unloaded all LoRA weights")
        except Exception as e:
            print(f"Error unloading LoRAs: {e}")
        
        self.lora_adapter_names = []
        self.current_lora_paths = {}
    
    def _load_checkpoint(self, checkpoint_path: str) -> StableDiffusionXLPipeline:
        """Load a full SDXL checkpoint with memory optimizations"""
        print("Loading with memory optimizations enabled...")
        
        load_kwargs = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
            "variant": "fp16" if self.dtype == torch.float16 else None,
            "low_cpu_mem_usage": True,
        }
        
        if checkpoint_path.endswith('.safetensors'):
            pipe = StableDiffusionXLPipeline.from_single_file(
                checkpoint_path,
                **load_kwargs
            )
        else:
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
        """Generate a single image"""
        if self.pipe is None:
            print("No model loaded for generation")
            return None
        
        try:
            generator = None
            if seed >= 0:
                generator = torch.Generator(device="cpu").manual_seed(seed)
            
            if clip_skip > 1 and hasattr(self.pipe, 'text_encoder'):
                self.pipe.text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
            
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
            
            image = result.images[0]
            del result
            self._clear_memory()
            
            return image
            
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM during generation. Attempting recovery...")
            self._clear_memory()
            
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
            return None