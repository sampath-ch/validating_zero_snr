import random
import os
import json
import torch
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# Unlock A100 Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class APODDataset(Dataset):
    def __init__(self, data_dir, tokenizer, size=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.entries =[]
        
        jsonl_path = os.path.join(data_dir, "train_metadata.jsonl")
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))
                    
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        
        # STRICTLY FLAT PATHING: Directly joins the folder path with the exact file name
        image_path = os.path.join(self.data_dir, entry["file_name"])
        
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image)
        
        text_inputs = self.tokenizer(
            entry["text"], padding="max_length", max_length=self.tokenizer.model_max_length, 
            truncation=True, return_tensors="pt"
        )
        return {"pixel_values": pixel_values, "input_ids": text_inputs.input_ids.squeeze(0)}


def train_lora():
    # SET TO THE IMPROVED ZERO-SNR MODEL
    model_id = "ByteDance/sd2.1-base-zsnr-laionaes5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16 

    # --- UPDATE 1: W&B Tracking for the Hero Run ---
    wandb.init(
        project="apod-diffusion-stress-test", 
        name="run-fixed-zsnr-10k-hero",
        config={"model": model_id, "batch_size": 32, "steps": 10000, "lr": 1.5e-4, "rank": 64}
    )

    print("Loading models into memory...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device, dtype=weight_dtype)
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Enable Gradient Checkpointing for 80GB
    unet.enable_gradient_checkpointing()

    print("Injecting LoRA layers...")
    # --- UPDATE 2: Much larger LoRA rank (r=64) to capture complex star structures ---
    lora_config = LoraConfig(
        r=64, lora_alpha=64, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    unet = get_peft_model(unet, lora_config)
    
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # 80GB Optimized DataLoader 
    dataset = APODDataset("/scratch/schettip/CS757/apod_dataset", tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=6, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    # --- UPDATE 3: 10,000 steps and lower learning rate (1.5e-4) ---
    max_steps = 10000
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=1.5e-4)
    
    # --- UPDATE 4: Increased Warmup Steps for stability ---
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=max_steps,
    )

    global_step = 0
    print("\nStarting Full 80GB Training Burn (10k Steps)...")
    unet.train()
    
    progress_bar = tqdm(total=max_steps, desc="Training Steps")
    
    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break
                
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # --- FIX: Conditioning Dropout for robust CFG ---
            if random.random() < 0.1:
                # Multiply the empty string by the batch size (bsz)
                empty_tokens = tokenizer(
                    [""] * bsz, 
                    padding="max_length", 
                    max_length=tokenizer.model_max_length, 
                    return_tensors="pt"
                ).input_ids.to(device)
                encoder_hidden_states = text_encoder(empty_tokens)[0]
            else:
                input_ids = batch["input_ids"].to(device)
                encoder_hidden_states = text_encoder(input_ids)[0]
            # -------------------------------------------------
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            with torch.autocast(device_type="cuda", dtype=weight_dtype):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()), 1.0)
            optimizer.step()
            lr_scheduler.step()

            # Live tracking to W&B Dashboard
            if global_step % 10 == 0:
                wandb.log({"train_loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step)
                
            # W&B Visual Validation every 500 steps
            if global_step % 500 == 0 and global_step > 0:
                print(f"\nGenerating W&B Validation Image at step {global_step}...")
                
                # FIX 3: Use DDIMScheduler for inference as stated in the paper
                from diffusers import DDIMScheduler
                val_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
                val_scheduler.config.timestep_spacing = "trailing"
                
                pipeline = StableDiffusionPipeline(
                    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, 
                    scheduler=val_scheduler, safety_checker=None, feature_extractor=None
                )
                pipeline.set_progress_bar_config(disable=True)
                
                with torch.autocast("cuda", dtype=weight_dtype), torch.no_grad():
                    image = pipeline(
                        "A majestic coronal loop of solar plasma against pitch black space", 
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        guidance_rescale=0.7  # <-- FIX 4: Algorithm 2 (CFG Rescale) from the paper
                    ).images[0]
                
                # Beam image directly to the cloud dashboard
                wandb.log({"Validation Image": wandb.Image(image, caption=f"Step {global_step}")}, step=global_step)
                
                del pipeline
                del val_scheduler
                torch.cuda.empty_cache()
                
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    progress_bar.close()
    print("\nTraining complete! Saving LoRA weights...")
    unet = unet.to(torch.float32)
    # --- UPDATE 5: Save to a new folder so you don't overwrite your 3k step test ---
    unet.save_pretrained("./lora_fixed_zsnr_output")
    wandb.finish()

if __name__ == "__main__":
    train_lora()