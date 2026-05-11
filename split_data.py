import os
import random

# Read all entries
with open("apod_dataset/verified_meta_data.jsonl", "r") as f:
    lines = f.readlines()

# Shuffle them so you get a random mix of space phenomena
random.seed(42)
random.shuffle(lines)

# Split 3000 for training, the rest (~400) for validation
train_lines = lines[:3000]
val_lines = lines[3000:]

# Save the new files
with open("apod_dataset/train_metadata.jsonl", "w") as f:
    f.writelines(train_lines)
    
with open("apod_dataset/val_metadata.jsonl", "w") as f:
    f.writelines(val_lines)

print(f"Created train set ({len(train_lines)} images) and val set ({len(val_lines)} images)")