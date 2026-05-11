import os
import json
import time
import google.generativeai as genai
from tqdm import tqdm

# --- CONFIGURATION ---
# Replace with your actual Gemini API Key
API_KEY = "YOUR_GEMINI_API_KEY"
INPUT_FILE = "apod_dataset/metadata.jsonl"
OUTPUT_FILE = "apod_dataset/verified_metadata.jsonl"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def condense_text(text):
    prompt = (
        f"Condense the following astronomical description into a concise caption "
        f"shorter than 65 words. Maintain the core visual details and scientific "
        f"essence, but remove filler words. Output ONLY the condensed text.\n\n"
        f"Description: {text}"
    )
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error calling API: {e}")
        return text  # Fallback to original if API fails

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    processed_entries = []
    
    # Read the JSONL entries
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"🚀 Found {len(lines)} entries. Starting the condensation process...")

    for line in tqdm(lines, desc="Processing Captions"):
        if not line.strip():
            continue
            
        data = json.loads(line)
        original_text = data.get("text", "")
        
        # Only condense if it's actually long; otherwise, save the API quota
        if len(original_text.split()) > 60:
            condensed = condense_text(original_text)
            data["text"] = condensed
        
        processed_entries.append(data)
        
        # Respect rate limits (Flash has generous limits, but small pauses help)
        time.sleep(0.5)

    # Write the new JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\n✅ Done! Condensed metadata saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()