import os
import json
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def create_qualitative_collage():
    csv_path = "./val_logs/validation_scores.csv"
    jsonl_path = "/scratch/schettip/CS757/apod_dataset/val_metadata.jsonl"
    
    dir_gt = "/scratch/schettip/CS757/apod_dataset"
    dir_fixed = "./val_logs/images_fixed"
    dir_flawed = "./val_logs/images_flawed"
    output_path = "./val_logs/plots/4_qualitative_collage.png"
    
    print("Loading data for qualitative collage...")
    df = pd.read_csv(csv_path)
    
    # Calculate the "Win Margin"
    # We want: High Fixed_Sim, Low Flawed_Sim AND Low Fixed_EMD, High Flawed_EMD
    df['Win_Margin'] = (df['Fixed_Sim'] - df['Flawed_Sim']) + (df['Flawed_EMD'] - df['Fixed_EMD'])
    
    # Sort and take the top 10 best showcases
    top_10 = df.sort_values(by='Win_Margin', ascending=False).head(10).reset_index(drop=True)
    
    # Load captions from JSONL
    captions = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                captions[data['file_name']] = data['text']
                
    # Create the Matplotlib Grid (3 Rows, 10 Columns)
    # Row 0: Ground Truth | Row 1: Fixed | Row 2: Flawed
    fig, axes = plt.subplots(3, 10, figsize=(25, 9))
    
    row_labels =["Ground Truth\n(Real Image)", "Fixed Zero-SNR\n", "Flawed Baseline\n(Gray Leak)"]
    for row in range(3):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=16, fontweight='bold', labelpad=15)
        
    for i, row_data in top_10.iterrows():
        fname = row_data['File']
        caption = captions.get(fname, "No caption found")
        
        # Truncate caption to the first 8 words
        words = caption.split()
        short_cap = " ".join(words[:8]) + "..."
        wrapped_cap = textwrap.fill(short_cap, width=25)
        
        # Load the 3 images, resize to square for a clean grid
        img_gt = Image.open(os.path.join(dir_gt, fname)).convert("RGB").resize((512, 512))
        img_fixed = Image.open(os.path.join(dir_fixed, fname)).convert("RGB").resize((512, 512))
        img_flawed = Image.open(os.path.join(dir_flawed, fname)).convert("RGB").resize((512, 512))
        
        # Plot them
        axes[0, i].imshow(img_gt)
        axes[1, i].imshow(img_fixed)
        axes[2, i].imshow(img_flawed)
        
        # Formatting (Remove axes ticks, add clean borders)
        for r in range(3):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
            for spine in axes[r, i].spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(2)
        
        # Add the caption under the bottom row
        axes[2, i].set_xlabel(wrapped_cap, fontsize=13, labelpad=10)
        
    plt.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.1) # Make room for row titles
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Qualitative Collage saved to {output_path}")

if __name__ == "__main__":
    create_qualitative_collage()