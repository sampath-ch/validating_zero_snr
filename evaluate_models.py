import os
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import wasserstein_distance
from skimage.color import rgb2lab
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import ToTensor
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image

# ---------------------------------------------------------
# 1. THE EVALUATOR CLASS (Updated: No Clipping)
# ---------------------------------------------------------
class WLPS_Evaluator:
    def __init__(self, device="cuda"):
        self.device = device
        print("\nLoading DINOv2 Perceptual Model...")
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.dino_model.eval()

    def get_luminance_emd(self, img_gt, img_gen):
        np_gt = (img_gt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        np_gen = (img_gen.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        lab_gt = rgb2lab(np_gt)
        lab_gen = rgb2lab(np_gen)
        
        l_gt = lab_gt[:, :, 0].flatten()
        l_gen = lab_gen[:, :, 0].flatten()
        
        emd = wasserstein_distance(l_gt, l_gen) / 100.0
        return emd

    def get_perceptual_similarity(self, img_gt, img_gen):
        inputs_gt = self.processor(images=img_gt, return_tensors="pt").to(self.device)
        inputs_gen = self.processor(images=img_gen, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            emb_gt = self.dino_model(**inputs_gt).last_hidden_state[:, 0, :] 
            emb_gen = self.dino_model(**inputs_gen).last_hidden_state[:, 0, :]
            
        sim = F.cosine_similarity(emb_gt, emb_gen, dim=-1).item()
        return max(0.0, sim) 

    def compute_wlps(self, img_gt, img_gen, alpha=2.0):
        s_percept = self.get_perceptual_similarity(img_gt, img_gen)
        w_1 = self.get_luminance_emd(img_gt, img_gen)
        
        # New clean formula without the clipping denominator
        wlps_score = s_percept / (1.0 + (alpha * w_1))
        
        return {
            "WLPS": wlps_score,
            "Similarity": s_percept,
            "Luminance_EMD": w_1
        }

# ---------------------------------------------------------
# 2. GENERATION HELPER
# ---------------------------------------------------------
# ---------------------------------------------------------
# 2. GENERATION HELPER
# ---------------------------------------------------------
def generate_images(val_data, output_dir, model_id, lora_path, is_fixed=False, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda"
    
    print(f"\nLoading Pipeline for {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    
    # --- FIX: Explicitly point diffusers to the PEFT adapter filename ---
    if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        pipe.load_lora_weights(lora_path, weight_name="adapter_model.safetensors")
    elif os.path.exists(os.path.join(lora_path, "adapter_model.bin")):
        pipe.load_lora_weights(lora_path, weight_name="adapter_model.bin")
    else:
        pipe.load_lora_weights(lora_path) # Fallback
    # --------------------------------------------------------------------
    
    pipe.set_progress_bar_config(disable=True)
    
    if is_fixed:
        print("Applying Trailing Timesteps fix for Zero-SNR model...")
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # Keep track of original indices to maintain deterministic seeding
    indexed_data = list(enumerate(val_data))

    print(f"Generating 400 images into {output_dir} (Batch Size: {batch_size})...")
    
    # Process in chunks of 'batch_size'
    for i in tqdm(range(0, len(indexed_data), batch_size), desc=f"Generating ({'Fixed' if is_fixed else 'Flawed'})"):
        batch = indexed_data[i : i + batch_size]
        
        # Filter out already generated images (allows safe resuming)
        batch_to_process =[]
        for idx, entry in batch:
            save_path = os.path.join(output_dir, entry["file_name"])
            if not os.path.exists(save_path):
                batch_to_process.append((idx, entry))
                
        if not batch_to_process:
            continue
            
        prompts = [entry["text"] for idx, entry in batch_to_process]
        filenames = [entry["file_name"] for idx, entry in batch_to_process]
        
        # CRITICAL: Create a unique but deterministic seed based on the image's original index.
        generators =[torch.Generator(device).manual_seed(1337 + idx) for idx, entry in batch_to_process]
        
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            if is_fixed:
                images = pipe(
                    prompts, 
                    num_inference_steps=30, 
                    generator=generators, 
                    guidance_rescale=0.7
                ).images
            else:
                images = pipe(
                    prompts, 
                    num_inference_steps=30, 
                    generator=generators
                ).images
                
        # Save generated batch
        for img, fn in zip(images, filenames):
            img.save(os.path.join(output_dir, fn))

    del pipe
    torch.cuda.empty_cache()

# ---------------------------------------------------------
# 3. MAIN EVALUATION LOOP
# ---------------------------------------------------------
def run_validation():
    dataset_dir = "/scratch/schettip/CS757/apod_dataset"
    val_jsonl = os.path.join(dataset_dir, "val_metadata.jsonl")
    flawed_out_dir = "./val_logs/images_flawed"
    fixed_out_dir = "./val_logs/images_fixed"
    
    val_data =[]
    with open(val_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                val_data.append(json.loads(line))

    # Phase 1 & 2: Generate Images
    generate_images(val_data, flawed_out_dir, "Manojb/stable-diffusion-2-1-base", "./lora_flawed_baseline_output", is_fixed=False)
    # Ensure this points to your new 10k "Hero Run" output folder!
    generate_images(val_data, fixed_out_dir, "ByteDance/sd2.1-base-zsnr-laionaes5", "./lora_fixed_zsnr_output", is_fixed=True)

    # Phase 3: Evaluate
    print("\nStarting WLPS Evaluation...")
    evaluator = WLPS_Evaluator(device="cuda")
    to_tensor = ToTensor()
    
    results =[]
    
    for entry in tqdm(val_data, desc="Evaluating WLPS"):
        filename = entry["file_name"]
        
        gt_path = os.path.join(dataset_dir, filename)
        flawed_path = os.path.join(flawed_out_dir, filename)
        fixed_path = os.path.join(fixed_out_dir, filename)
        
        try:
            img_gt = to_tensor(Image.open(gt_path).convert("RGB").resize((512, 512)))
            img_flawed = to_tensor(Image.open(flawed_path).convert("RGB"))
            img_fixed = to_tensor(Image.open(fixed_path).convert("RGB"))
        except Exception as e:
            print(f"Skipping {filename} due to loading error: {e}")
            continue

        flawed_metrics = evaluator.compute_wlps(img_gt, img_flawed)
        fixed_metrics = evaluator.compute_wlps(img_gt, img_fixed)
        
        results.append({
            "File": filename,
            "Flawed_WLPS": flawed_metrics["WLPS"],
            "Fixed_WLPS": fixed_metrics["WLPS"],
            "Flawed_EMD": flawed_metrics["Luminance_EMD"],
            "Fixed_EMD": fixed_metrics["Luminance_EMD"],
            "Flawed_Sim": flawed_metrics["Similarity"],
            "Fixed_Sim": fixed_metrics["Similarity"]
        })

    csv_path = "./val_logs/validation_scores.csv"
    os.makedirs("./val_logs", exist_ok=True)
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    avg_flawed_wlps = sum(r["Flawed_WLPS"] for r in results) / len(results)
    avg_fixed_wlps = sum(r["Fixed_WLPS"] for r in results) / len(results)
    
    print("\n" + "="*40)
    print("FINAL EVALUATION RESULTS (Averages)")
    print("="*40)
    print(f"Baseline (Flawed) WLPS : {avg_flawed_wlps:.4f}")
    print(f"Zero-SNR (Fixed) WLPS  : {avg_fixed_wlps:.4f}")
    print(f"\nDetailed logs saved to: {csv_path}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_validation()