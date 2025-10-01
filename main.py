#!/usr/bin/env python3
"""
SDXL XY Plot Generator - Main entry point with CLI override support
"""

import argparse
import sys
import toml
import re
from pathlib import Path
from typing import Dict, Any
from xy_plot_generator import FlexibleXYPlotGenerator


def parse_override(override_str: str) -> tuple:
    """
    Parse an override string in the format 'section.key=value' or 'section.subsection.key=value'
    Returns (path_list, value) where path_list is ['section', 'key'] or ['section', 'subsection', 'key']
    """
    if '=' not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Expected 'section.key=value'")
    
    path, value = override_str.split('=', 1)
    path_parts = path.split('.')
    
    if len(path_parts) < 2:
        raise ValueError(f"Invalid override path: {path}. Expected at least 'section.key'")
    
    # Try to parse the value as different types
    value = value.strip()
    
    # Check for boolean
    if value.lower() in ['true', 'false']:
        value = value.lower() == 'true'
    # Check for integer
    elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
        value = int(value)
    # Check for float
    elif re.match(r'^-?\d+\.?\d*$', value):
        value = float(value)
    # Check for list (simple format: [val1,val2,val3])
    elif value.startswith('[') and value.endswith(']'):
        list_content = value[1:-1]
        if list_content:
            # Split by comma and strip whitespace
            list_items = [item.strip() for item in list_content.split(',')]
            # Remove quotes if present
            list_items = [item.strip('"').strip("'") for item in list_items]
            value = list_items
        else:
            value = []
    # Otherwise keep as string, removing quotes if present
    else:
        value = value.strip('"').strip("'")
    
    return path_parts, value


def apply_override(config: Dict[str, Any], path_parts: list, value: Any) -> None:
    """
    Apply an override to the config dictionary
    """
    # Navigate to the correct level
    current = config
    for part in path_parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    
    # Set the value
    current[path_parts[-1]] = value


def load_config_with_overrides(config_path: str, overrides: list) -> Dict[str, Any]:
    """
    Load config file and apply CLI overrides
    """
    # Load base configuration
    with open(config_path, 'r') as f:
        config = toml.load(f)
    
    # Apply overrides
    for override_str in overrides:
        try:
            path_parts, value = parse_override(override_str)
            apply_override(config, path_parts, value)
            print(f"Applied override: {'.'.join(path_parts)} = {value}")
        except Exception as e:
            print(f"Warning: Failed to apply override '{override_str}': {e}")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate XY plot grids for SDXL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Override examples:
  --set checkpoint.path=/path/to/model.safetensors
  --set checkpoint.values=[/path/to/model1.safetensors,/path/to/model2.safetensors]
  --set generation.steps=50
  --set generation.cfg_scale=8.5
  --set xy_plot.x_axis=prompt
  --set xy_plot.y_axis=checkpoint
  --set prompt.values=[portrait,landscape,abstract]
  --set lora1.path=/path/to/lora.safetensors
  --set lora1.weight=0.8
  --set output.output_dir=./my_output
  --set memory_optimization.enable_cpu_offload=false
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.toml",
        help="Path to configuration file (default: config.toml)"
    )
    
    parser.add_argument(
        "--set", "-s",
        action="append",
        dest="overrides",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a config value (can be used multiple times)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the final configuration and exit without generating"
    )
    
    parser.add_argument(
        "--save-config",
        type=str,
        metavar="PATH",
        help="Save the final configuration (with overrides) to a new file"
    )
    
    args = parser.parse_args()
    
    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Load config with overrides
    try:
        config = load_config_with_overrides(str(config_path), args.overrides)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Save config if requested
    if args.save_config:
        save_path = Path(args.save_config)
        with open(save_path, 'w') as f:
            toml.dump(config, f)
        print(f"Configuration saved to: {save_path}")
    
    # Dry run - just print config
    if args.dry_run:
        print("\nFinal configuration:")
        print("=" * 50)
        print(toml.dumps(config))
        print("=" * 50)
        sys.exit(0)
    
    # Create temporary config file with overrides for the generator
    # (This is necessary because the generator expects a file path)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as tmp_config:
        toml.dump(config, tmp_config)
        tmp_config_path = tmp_config.name
    
    # Run generator
    try:
        print(f"Using configuration: {config_path}")
        if args.overrides:
            print(f"With {len(args.overrides)} override(s)")
        
        generator = FlexibleXYPlotGenerator(tmp_config_path)
        generator.generate_grid()
        print("\nGeneration complete!")
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temporary config file
        import os
        if 'tmp_config_path' in locals():
            try:
                os.unlink(tmp_config_path)
            except:
                pass


if __name__ == "__main__":
    main()