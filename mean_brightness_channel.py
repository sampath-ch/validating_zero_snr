import torch
import numpy as np
from diffusers import UNet2DConditionModel, DDPMScheduler
import matplotlib.pyplot as plt

# 1. Define the Hooking Class to peek inside the Black Box
class LayerHook:
    def __init__(self, module):
        self.activations = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # The mid_block outputs a tensor, we clone and detach it to analyze later
        self.activations = output.clone().detach()

    def remove(self):
        self.hook.remove()

def run_interpretability_probe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    models = {
        "Flawed": "Manojb/stable-diffusion-2-1-base",
        "Fixed (Zero-SNR)": "ByteDance/sd2.1-base-zsnr-laionaes5"
    }

    # We will test shifts in the input noise mean from -0.1 (dark) to +0.1 (bright)
    mean_shifts = np.linspace(-0.1, 0.1, 11)
    
    results = {
        "Flawed": {"mid_block_means": [], "x0_pred_means":[]},
        "Fixed (Zero-SNR)": {"mid_block_means": [], "x0_pred_means":[]}
    }

    # Generate the EXACT same base noise to ensure a perfectly fair test
    # torch.manual_seed(42)
    base_noise = torch.randn(1, 4, 64, 64).to(device, dtype=torch.float16)
    
    # Dummy text embeddings (empty prompt)
    encoder_hidden_states = torch.zeros((1, 77, 1024)).to(device, dtype=torch.float16)
    
    timestep = torch.tensor([999]).to(device)

    for model_name, model_id in models.items():
        print(f"\nProbing {model_name} Model...")
        
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16).to(device)
        scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Attach the hook to the UNet's mid_block (the deepest bottleneck)
        mid_block_hook = LayerHook(unet.mid_block)
        
        alpha_bar_T = scheduler.alphas_cumprod[999].item()

        for shift in mean_shifts:
            # INTERVENTION: Artificially shift the mean of the input noise
            manipulated_noise = base_noise + shift
            
            with torch.no_grad():
                # Forward pass
                model_output = unet(manipulated_noise, timestep, encoder_hidden_states).sample
                
            # Read the stethoscope (mid_block activations)
            mid_act = mid_block_hook.activations
            results[model_name]["mid_block_means"].append(mid_act.mean().item())
            
            # Reconstruct what the model *thinks* the original image (x0) looks like
            if scheduler.config.prediction_type == "epsilon":
                # Flawed model predicts noise. Algebra to find x0:
                x0_pred = (manipulated_noise - (1 - alpha_bar_T)**0.5 * model_output) / (alpha_bar_T**0.5)
            elif scheduler.config.prediction_type == "v_prediction":
                # Fixed model predicts velocity. Algebra to find x0:
                x0_pred = (alpha_bar_T**0.5) * manipulated_noise - ((1 - alpha_bar_T)**0.5) * model_output
            
            results[model_name]["x0_pred_means"].append(x0_pred.mean().item())

        mid_block_hook.remove()
        del unet, scheduler
        torch.cuda.empty_cache()

    # --- Print the Analytics ---
    print("\n" + "="*50)
    print("MECHANISTIC INTERPRETABILITY RESULTS")
    print("="*50)
    
    for model_name in models.keys():
        # Calculate the slopes (Sensitivity)
        mid_slope = (results[model_name]["mid_block_means"][-1] - results[model_name]["mid_block_means"][0]) / 0.2
        x0_slope = (results[model_name]["x0_pred_means"][-1] - results[model_name]["x0_pred_means"][0]) / 0.2
        
        print(f"\n{model_name} Model:")
        print(f"  Mid-Block Sensitivity to Noise Shift: {mid_slope:.4f}")
        print(f"  Final Image Brightness Sensitivity:   {x0_slope:.4f}x Multiplier")
        

if __name__ == "__main__":
    run_interpretability_probe()