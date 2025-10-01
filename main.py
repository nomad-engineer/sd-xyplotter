#!/usr/bin/env python3
"""
SDXL XY Plot Generator - Main entry point
"""

import argparse
import toml
import sys
from pathlib import Path
from sdxl_generator import XYPlotGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate XY plot grids for SDXL models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to configuration file (default: config.toml)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Override model directory from config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Override prompts file from config"
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        help="Override base checkpoint for LoRA mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # Override config with command line arguments
    if args.model_dir:
        config['paths']['model_dir'] = args.model_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if args.prompts_file:
        config['paths']['prompts_file'] = args.prompts_file
    if args.base_checkpoint:
        config['paths']['base_checkpoint'] = args.base_checkpoint
    
    # Validate paths
    model_dir = Path(config['paths']['model_dir'])
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
        
    prompts_file = Path(config['paths']['prompts_file'])
    if not prompts_file.exists():
        print(f"Error: Prompts file not found: {prompts_file}")
        sys.exit(1)
    
    # Print configuration
    print("Configuration:")
    print(f"  Model directory: {config['paths']['model_dir']}")
    print(f"  Output directory: {config['paths']['output_dir']}")
    print(f"  Prompts file: {config['paths']['prompts_file']}")
    if config['paths'].get('base_checkpoint'):
        print(f"  Base checkpoint (LoRA mode): {config['paths']['base_checkpoint']}")
    print(f"  Resolution: {config['generation']['resolution']}")
    print(f"  Steps: {config['generation']['steps']}")
    print(f"  CFG Scale: {config['generation']['cfg_scale']}")
    print(f"  Seed: {config['generation']['seed']}")
    print()
    
    # Run generator
    try:
        generator = XYPlotGenerator(config)
        generator.generate_grid()
        print("\nGeneration complete!")
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()