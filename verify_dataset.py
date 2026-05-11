import json
import os
from transformers import CLIPTokenizer

def verify_sd_dataset(input_file, output_file, image_folder):
    # Initialize the standard tokenizer used by Stable Diffusion (CLIP ViT-L/14)
    print("Loading CLIP Tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    # Stable Diffusion's hard limit is 77 tokens (including start/end tokens)
    MAX_TOKENS = 77 
    
    seen_files = set()
    valid_entries = []
    
    duplicate_count = 0
    missing_count = 0
    oversized_count = 0

    print("Starting verification...\n")
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue
                
            entry = json.loads(line.strip())
            file_name = entry.get("file_name")
            text = entry.get("text", "")

            # --- Check 1: Duplicates ---
            if file_name in seen_files:
                print(f"[Line {line_num}] Duplicate found and skipped: {file_name}")
                duplicate_count += 1
                continue
            
            # --- Check 2: File Existence ---
            image_path = os.path.join(image_folder, file_name)
            if not os.path.isfile(image_path):
                print(f"[Line {line_num}] Image missing from folder: {file_name}")
                missing_count += 1
                continue
                
            # --- Check 3: CLIP Token Limit ---
            # Tokenize the text to get the actual token count
            tokens = tokenizer(text)["input_ids"]
            token_count = len(tokens)
            
            if token_count > MAX_TOKENS:
                print(f"[Line {line_num}] Caption too long ({token_count}/77 tokens): {file_name}")
                oversized_count += 1
                continue
                
            # If all checks pass, record the file as seen and store the entry
            seen_files.add(file_name)
            valid_entries.append(entry)

    # --- Final Output Creation ---
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in valid_entries:
            outfile.write(json.dumps(entry) + '\n')

    # Summary Report
    print("\n--- Verification Summary")
    print(f"Total processed: {line_num}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Missing images: {missing_count}")
    print(f"Oversized captions: {oversized_count}")
    print(f"Valid entries saved: {len(valid_entries)}")
    print(f"Output saved to: {output_file}")


# Run the script

if __name__ == "__main__":
    # Define your paths here
    INPUT_JSONL = "/scratch/schettip/CS757/apod_dataset/meta_data.jsonl"
    OUTPUT_JSONL = "/scratch/schettip/CS757/apod_dataset/verified_meta_data.jsonl"
    IMAGE_DIRECTORY = "/scratch/schettip/CS757/apod_dataset/" # Update this to your folder path
    
    verify_sd_dataset(INPUT_JSONL, OUTPUT_JSONL, IMAGE_DIRECTORY)