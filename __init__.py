import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
from PIL import Image

class SigmaVisualizerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
            },
            "optional": {
                "title": ("STRING", {"default": "Sigma Values Distribution"}),
                "width": ("INT", {"default": 800, "min": 400, "max": 2048, "step": 50}),
                "height": ("INT", {"default": 600, "min": 300, "max": 1536, "step": 50}),
                "show_grid": ("BOOLEAN", {"default": True}),
                "plot_type": (["line", "scatter", "histogram", "both"], {"default": "line"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph_image",)
    FUNCTION = "visualize_sigmas"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def visualize_sigmas(self, sigmas, title="Sigma Values Distribution", width=800, height=600, show_grid=True, plot_type="line"):
        # Convert to numpy for plotting
        sigmas_np = sigmas.cpu().numpy() if isinstance(sigmas, torch.Tensor) else np.array(sigmas)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        if plot_type == "line" or plot_type == "both":
            ax.plot(range(len(sigmas_np)), sigmas_np, 'b-', linewidth=1.5, label='Sigma Values', alpha=0.8)
        
        if plot_type == "scatter" or plot_type == "both":
            ax.scatter(range(len(sigmas_np)), sigmas_np, c='red', s=20, alpha=0.6, label='Sigma Points')
        
        if plot_type == "histogram":
            ax.hist(sigmas_np, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Sigma Value')
            ax.set_ylabel('Frequency')
        else:
            ax.set_xlabel('Step Index')
            ax.set_ylabel('Sigma Value')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        if plot_type == "both":
            ax.legend()
        
        # Add statistics text
        if plot_type != "histogram":
            stats_text = f'Min: {sigmas_np.min():.4f}\nMax: {sigmas_np.max():.4f}\nMean: {sigmas_np.mean():.4f}\nStd: {sigmas_np.std():.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        plt.close(fig)
        
        # Convert to PIL Image and then to tensor format expected by ComfyUI
        pil_image = Image.open(buf)
        image_np = np.array(pil_image)
        
        # Convert to the format expected by ComfyUI (batch, height, width, channels)
        # Ensure we have RGB channels
        if image_np.shape[-1] == 4:  # RGBA
            image_np = image_np[:, :, :3]  # Drop alpha channel
        
        # Normalize to [0, 1] range and add batch dimension
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return (image_tensor,)


class CFGScheduleVisualizerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cfg_scaling_sigmas": ("SIGMAS",),
                "cfg_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "cfg_max": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "control_sigmas": ("SIGMAS",),
                "title": ("STRING", {"default": "CFG Schedule"}),
                "width": ("INT", {"default": 800, "min": 400, "max": 2048, "step": 50}),
                "height": ("INT", {"default": 600, "min": 300, "max": 1536, "step": 50}),
                "show_grid": ("BOOLEAN", {"default": True}),
                "show_control_overlay": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cfg_schedule_image",)
    FUNCTION = "visualize_cfg_schedule"
    CATEGORY = "sampling/custom_sampling/sigmas"

    def visualize_cfg_schedule(self, cfg_scaling_sigmas, cfg_min, cfg_max, control_sigmas=None, title="CFG Schedule", width=800, height=600, show_grid=True, show_control_overlay=True):
        # Convert sigmas to numpy
        sigmas_np = cfg_scaling_sigmas.cpu().numpy() if isinstance(cfg_scaling_sigmas, torch.Tensor) else np.array(cfg_scaling_sigmas)
        
        # Remove the final 0.0 sigma for calculation purposes
        if len(sigmas_np) > 1 and sigmas_np[-1] == 0.0:
            calc_sigmas = sigmas_np[:-1]
        else:
            calc_sigmas = sigmas_np
        
        # Calculate sigma min/max from the sequence
        sigma_min = calc_sigmas.min()
        sigma_max = calc_sigmas.max()
        
        # Calculate CFG values for each timestep using the ScheduledGuider logic
        # Note: High sigma = early steps (want cfg_min), Low sigma = late steps (want cfg_max)
        cfg_values = []
        for sigma in calc_sigmas:
            # Calculate current percentage based on sigma progression (0=low sigma/end, 1=high sigma/start)
            current_percent = ((sigma - sigma_min) / (sigma_max - sigma_min))
            current_cfg = ((cfg_max - cfg_min) * current_percent + cfg_min)
            cfg_values.append(current_cfg)
        
        cfg_values = np.array(cfg_values)
        
        # Create the plot
        fig, ax1 = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Plot CFG schedule
        color_cfg = 'tab:blue'
        ax1.set_xlabel('Step Index')
        ax1.set_ylabel('CFG Value', color=color_cfg)
        line1 = ax1.plot(range(len(cfg_values)), cfg_values, 'b-', linewidth=2, label='CFG Schedule', alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color_cfg)
        ax1.set_ylim(min(cfg_min, cfg_max) * 0.9, max(cfg_min, cfg_max) * 1.1)
        
        # Handle control sigma overlay
        lines = line1
        labels = [l.get_label() for l in lines]
        
        if show_control_overlay and control_sigmas is not None:
            ax2 = ax1.twinx()
            color_sigma = 'tab:orange'
            ax2.set_ylabel('Sigma Value', color=color_sigma)
            ax2.tick_params(axis='y', labelcolor=color_sigma)
            
            control_sigmas_np = control_sigmas.cpu().numpy() if isinstance(control_sigmas, torch.Tensor) else np.array(control_sigmas)
            # Remove final 0.0 if present
            if len(control_sigmas_np) > 1 and control_sigmas_np[-1] == 0.0:
                control_calc_sigmas = control_sigmas_np[:-1]
            else:
                control_calc_sigmas = control_sigmas_np
            
            line2 = ax2.plot(range(len(control_calc_sigmas)), control_calc_sigmas, 's--', color='purple', linewidth=1, alpha=0.7, markersize=2, label='Control Sigmas')
            lines = lines + line2
            labels = [l.get_label() for l in lines]
        
        # Show legend if we have multiple lines
        if len(lines) > 1:
            ax1.legend(lines, labels, loc='upper left')
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        if show_grid:
            ax1.grid(True, alpha=0.3)
        
        # Add statistics text with debugging info
        first_cfg = cfg_values[0] if len(cfg_values) > 0 else 0
        last_cfg = cfg_values[-1] if len(cfg_values) > 0 else 0
        first_sigma = calc_sigmas[0] if len(calc_sigmas) > 0 else 0
        last_sigma = calc_sigmas[-1] if len(calc_sigmas) > 0 else 0
        stats_text = f'CFG: {cfg_min:.2f} → {cfg_max:.2f}\nσ Range: {sigma_min:.4f} → {sigma_max:.4f}\nFirst: σ={first_sigma:.4f}, CFG={first_cfg:.2f}\nLast: σ={last_sigma:.4f}, CFG={last_cfg:.2f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        plt.close(fig)
        
        # Convert to PIL Image and then to tensor format expected by ComfyUI
        pil_image = Image.open(buf)
        image_np = np.array(pil_image)
        
        # Convert to the format expected by ComfyUI (batch, height, width, channels)
        # Ensure we have RGB channels
        if image_np.shape[-1] == 4:  # RGBA
            image_np = image_np[:, :, :3]  # Drop alpha channel
        
        # Normalize to [0, 1] range and add batch dimension
        image_tensor = torch.from_numpy(image_np).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return (image_tensor,)


NODE_CLASS_MAPPINGS = {
    "SigmaVisualizerNode": SigmaVisualizerNode,
    "CFGScheduleVisualizerNode": CFGScheduleVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SigmaVisualizerNode": "Sigma Visualizer",
    "CFGScheduleVisualizerNode": "CFG Schedule Visualizer",
}