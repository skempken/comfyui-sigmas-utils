# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ComfyUI custom nodes package focused on understanding and visualizing sigma values in diffusion processes. It provides visualization tools for sigma schedules and CFG scheduling based on sigma progressions. Sigma values control the noise level at each step of the denoising process, making them crucial for controlling generation quality and style.

## ComfyUI Custom Node Architecture

### Node Structure
- Each node is a Python class with specific methods:
  - `INPUT_TYPES()`: Class method defining input parameters and types
  - `FUNCTION`: String specifying the entry point method name
  - `RETURN_TYPES`: Tuple of output types
  - `CATEGORY`: UI category placement
  - Main function (named in FUNCTION attribute) that processes inputs

### Required Elements
- `__init__.py` file containing node classes
- `NODE_CLASS_MAPPINGS` dictionary mapping node names to classes
- Optional `NODE_DISPLAY_NAME_MAPPINGS` for friendly names

### Common Input Types
- `MODEL`: ComfyUI model object
- `SIGMAS`: Sigma schedule tensor
- `FLOAT`: Numeric parameters with min/max/step/default
- `INT`: Integer parameters
- `STRING`: Text inputs

### Sigma-Specific Knowledge
- Sigmas are torch tensors representing noise levels
- Higher sigmas = more noise (early denoising steps)
- Lower sigmas = less noise (late denoising steps)  
- Final sigma is always 0.0
- Access model sigmas via `model.get_model_object("model_sampling")`

## Development Commands

ComfyUI automatically loads custom nodes from the `custom_nodes` directory. No build process required - changes are picked up on restart.

### Testing
- Restart ComfyUI to reload node changes
- Use ComfyUI's workflow interface to test nodes
- Check terminal output for Python errors/logging

### Node Categories
Use these standard categories for sigma-related nodes:
- `"sampling/custom_sampling/schedulers"` for scheduler nodes
- `"sampling/custom_sampling/sigmas"` for sigma manipulation
- `"utils"` for general utilities

## Current Nodes

### SigmaVisualizerNode
- **Purpose**: Visualizes sigma value distributions
- **Inputs**: SIGMAS, optional plot configuration
- **Outputs**: IMAGE (matplotlib plot)
- **Features**: Line, scatter, histogram, or combined plots with statistics

### CFGScheduleVisualizerNode  
- **Purpose**: Visualizes CFG scheduling based on sigma progression
- **Inputs**: cfg_scaling_sigmas, cfg_min, cfg_max, optional control_sigmas
- **Outputs**: IMAGE (dual-axis plot)
- **Features**: Shows CFG interpolation curve with optional control sigma overlay
- **CFG Logic**: `current_cfg = ((cfg_max - cfg_min) * current_percent + cfg_min)`
  where `current_percent = ((sigma - sigma_min) / (sigma_max - sigma_min))`

## Project Structure
```
comfyui-sigmas-utils/
├── __init__.py          # Main node definitions
├── CLAUDE.md           # This file  
└── README.md           # User documentation
```

## Key Implementation Patterns
- Use matplotlib with 'Agg' backend for image generation
- Convert plots to PIL → numpy → torch tensors for ComfyUI
- Handle optional sigma inputs gracefully (check for None)
- Remove final 0.0 sigma for calculations when present
- Use dual-axis plots for CFG+sigma visualization
- Provide meaningful debug statistics in plot annotations