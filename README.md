# Validating Zero-SNR in Stable Diffusion

This repository contains the data pipeline, training scripts, and evaluation metrics required to validate the Zero-SNR implementation in Stable Diffusion 2.1. The project compares a standard (flawed) baseline model against an improved Zero-SNR model, measuring fidelity and luminance accuracy using the WLPS metric, and includes a mechanistic probe to analyze the models' sensitivity to noise bias.

## Environment Setup

This project is configured to run on a high-performance computing cluster (e.g., Slurm) equipped with NVIDIA A100 GPUs. 

To set up the environment, you must load the appropriate CUDA and cuDNN modules, and create a virtual environment. Run the following commands in your terminal:

```bash
# 1. Load the required modules
module load python/3.10
module load cuda/12.6
module load cudnn/9.6.0.74-12.6

# 2. Create and activate a Python virtual environment
python -m venv lora_env
source lora_env/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt
```

> **Note:** You will also need to authenticate with Weights & Biases if you plan to track the training runs. The provided `.slurm` files handle this via an environment variable.

---

## Pipeline Execution Order

To reproduce the experiments, execute the scripts in the following sequential order.

### Phase 1: Data Preparation

We utilize the APOD (Astronomy Picture of the Day) dataset to stress-test the models on high-contrast, deep-space imagery.

1. **Download Data:** Run `python get_data.py` to asynchronously download the dataset using the Hugging Face `datasets` library. This generates the initial `metadata.jsonl`.

2. **Condense Captions:** Run `python condense_captions.py`. Stable Diffusion's CLIP tokenizer has a strict 77-token limit, and many raw NASA descriptions are far too verbose. This script uses the Gemini 1.5 Flash API to intelligently summarize long descriptions into concise captions (under 70 words) while preserving essential visual and scientific details.

3. **Verify Integrity:** Run `python verify_dataset.py`. This script sanitizes the data by removing duplicates, checking for missing image files, and performing a final check to ensure all captions — including those processed by the LLM — fit within the 77-token limit.

4. **Partition Data:** Run `python split_data.py` to shuffle and split the verified dataset into discrete training (~3000 images) and validation (~400 images) sets.

---

### Phase 2: Model Training

We fine-tune the models using Low-Rank Adaptation (LoRA).

> **Note:** `train_text_to_image_lora.py` is included purely as the foundational blueprint and reference for the customized training loops; you do not need to run it directly.

1. **Flawed Baseline Training:** Submit the first Slurm job by running `sbatch run.slurm`. This executes `base_train.py`, which fine-tunes the standard `stable-diffusion-2-1-base` model.

2. **Improved Zero-SNR Training:** Submit the second Slurm job by running `sbatch run2.slurm`. This executes `improved_train.py`, which fine-tunes the updated `sd2.1-base-zsnr-laionaes5` model and applies trailing timesteps alongside an explicit CFG rescale.

---

### Phase 3: Validation and Evaluation

Once both models are trained and their respective LoRA weights are saved, generate and score the validation images.

1. **Evaluate Models:** Run `python evaluate_models.py`. This script handles inference for both the baseline and the fixed model across the validation set. It then computes the **LFI (Luminance Fidelity Index)**, which tracks both the Perceptual Similarity (DINOv2) and the Luminance EMD (Earth Mover's Distance). The results are saved to `./val_logs/validation_scores.csv`.

---

### Phase 4: Analysis and Visualization

Generate statistical plots and visual grids to interpret the evaluation metrics.

1. **Statistical Plots:** Run `python analysis_val_logs.py`. This will read the `.csv` scores and output three graphs into `./val_logs/plots/`:
   - A component breakdown bar chart.
   - A kernel density estimate plotting the darkness vs. fidelity trade-off.
   - An image-by-image WLPS delta chart tracking pairwise improvements.

2. **Qualitative Grids:** Run `python generate_collage.py` to extract the top 10 images with the highest "win margin" and assemble them into a side-by-side grid comparing the Ground Truth, the Fixed Zero-SNR, and the Flawed Baseline.

---

### Phase 5: Mechanistic Interpretability

To understand why the models behave differently, we use a diagnostic probe to monitor the internal network activations.

1. **Run Diagnostic Probe:** Execute `python mean_brightness_channel.py`. This script intervenes in the forward pass by applying an artificial mean shift to the pure input noise. It hooks into the UNet's mid-block to measure activation sensitivity and calculates the multiplier effect on the predicted initial image brightness (x0).
