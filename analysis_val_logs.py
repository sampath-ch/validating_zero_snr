import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot():
    csv_path = "./val_logs/validation_scores.csv"
    output_dir = "./val_logs/plots"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading CSV data...")
    df = pd.read_csv(csv_path)

    sns.set_theme(style="whitegrid", palette="muted")

    # PLOT 1: The Component Breakdown (Bar Chart)
    
    plt.figure(figsize=(8, 6))
    
    # Calculate means (No more Clipping Penalty)
    means = {
        'Metric Component':['Perceptual Similarity\n(Higher = Better)', 
                             'Luminance EMD\n(Lower = Better)'],
        'Flawed Baseline': [df['Flawed_Sim'].mean(), df['Flawed_EMD'].mean()],
        'Fixed Zero-SNR': [df['Fixed_Sim'].mean(), df['Fixed_EMD'].mean()]
    }
    mean_df = pd.DataFrame(means).melt(id_vars='Metric Component', var_name='Model', value_name='Average Score')
    
    sns.barplot(data=mean_df, x='Metric Component', y='Average Score', hue='Model')
    plt.title("Deconstructing the LFI Score Components", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_component_breakdown.png"), dpi=300)
    plt.close()

    # PLOT 2: The Darkness vs Fidelity Trade-off
    
    plt.figure(figsize=(10, 8))
    
    flawed_data = pd.DataFrame({'EMD': df['Flawed_EMD'], 'Similarity': df['Flawed_Sim'], 'Model': 'Flawed Baseline'})
    fixed_data = pd.DataFrame({'EMD': df['Fixed_EMD'], 'Similarity': df['Fixed_Sim'], 'Model': 'Fixed Zero-SNR'})
    combined = pd.concat([flawed_data, fixed_data])

    sns.kdeplot(data=combined, x="EMD", y="Similarity", hue="Model", fill=True, alpha=0.5, levels=5)
    plt.title("Fidelity vs. Tone: The Diffusion Trade-off", fontsize=14, pad=15)
    plt.xlabel("Luminance EMD (Distance from True Brightness) → Lower is better")
    plt.ylabel("DINOv2 Perceptual Similarity → Higher is better")
    
    plt.annotate("Ideal Zone\n(High Sim, Low EMD)", xy=(combined['EMD'].min(), combined['Similarity'].max()), 
                 xytext=(combined['EMD'].min() + 0.05, combined['Similarity'].max() - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_fidelity_vs_tone_tradeoff.png"), dpi=300)
    plt.close()

    # PLOT 3: Pairwise Difference Tracking
    
    plt.figure(figsize=(12, 6))
    
    df['LFI_Delta'] = df['Fixed_LFI'] - df['Flawed_LFI']
    df_sorted = df.sort_values('LFI_Delta').reset_index(drop=True)
    
    colors =['#d62728' if x < 0 else '#2ca02c' for x in df_sorted['LFI_Delta']]
    
    plt.bar(df_sorted.index, df_sorted['LFI_Delta'], color=colors, width=1.0)
    plt.axhline(0, color='black', linewidth=1.5)
    plt.title("Image-by-Image LFI Change (Fixed - Flawed)", fontsize=14, pad=15)
    plt.xlabel("Validation Images (Ranked by Delta)")
    plt.ylabel("Change in Score (Positive = Fixed Model won)")
    
    wins = (df['LFI_Delta'] > 0).sum()
    losses = (df['LFI_Delta'] < 0).sum()
    
    # Adjusted text position to dynamically fit the chart
    y_max = df['LFI_Delta'].max()
    y_min = df['LFI_Delta'].min()
    plt.text(len(df)*0.05, y_max * 0.8, f"Fixed model improved {wins} images", color='green', fontsize=12, fontweight='bold')
    plt.text(len(df)*0.05, y_min * 0.8, f"Flawed model won on {losses} images", color='red', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_image_by_image_delta.png"), dpi=300)
    plt.close()

    print(f" Success! 3 analytical plots saved to {output_dir}")
    
    print("\n--- QUICK DIAGNOSTICS ---")
    print(f"Flawed Avg LFI: {df['Flawed_LFI'].mean():.4f} | Fixed Avg LFI: {df['Fixed_LFI'].mean():.4f}")
    print(f"Flawed Avg EMD:  {df['Flawed_EMD'].mean():.4f} | Fixed Avg EMD:  {df['Fixed_EMD'].mean():.4f}")
    print(f"Flawed Avg Sim:  {df['Flawed_Sim'].mean():.4f} | Fixed Avg Sim:  {df['Fixed_Sim'].mean():.4f}")

if __name__ == "__main__":
    analyze_and_plot()