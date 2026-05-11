import os
import json
import asyncio
import aiohttp
from datasets import load_dataset
from tqdm.asyncio import tqdm
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Set this to your Hopper scratch directory
SAVE_DIR = "./apod_dataset"
METADATA_FILE = os.path.join(SAVE_DIR, "metadata.jsonl")
os.makedirs(SAVE_DIR, exist_ok=True)

async def download_image(session, url, file_name, text, metadata_list):
    try:
        # Standardize headers to prevent NASA server rejections
        headers = {'User-Agent': 'Mozilla/5.0'}
        async with session.get(url, headers=headers, timeout=15) as response:
            if response.status == 200:
                content = await response.read()
                with open(os.path.join(SAVE_DIR, file_name), "wb") as f:
                    f.write(content)
                # diffusers expects 'file_name' and 'text' keys
                metadata_list.append({"file_name": file_name, "text": text})
    except Exception:
        # Drop dead links silently to keep the pipeline moving
        pass 

async def main():
    print("Loading AstroLLaVA_convos dataset...")
    dataset = load_dataset("UniverseTBD/AstroLLaVA_convos", split="train")
    
    # FIX: Drop the image column so the library doesn't try to decode corrupted files.
    # We only need the text and URLs since we download the images ourselves.
    if "image" in dataset.column_names:
        dataset = dataset.remove_columns("image")
        
    print("Filtering for APOD candidates...")
    # Now the filter will run safely without hitting PIL errors
    apod_data = dataset.filter(lambda x: "apod" in str(x.get("corpus", "")).lower() or "apod.nasa.gov" in str(x.get("image_url", "")))
    
    print(f"Found {len(apod_data)} APOD candidates. Starting async download...")
    
    metadata_list = []
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for idx, row in enumerate(apod_data):
            url = row.get("image_url") or row.get("url")
            text = row.get("caption") or row.get("text") or row.get("explanation")
            
            if url and text:
                if url.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_name = f"apod_{idx:05d}.jpg"
                    tasks.append(download_image(session, url, file_name, text, metadata_list))
                
        await tqdm.gather(*tasks)
        
    print(f"Successfully downloaded {len(metadata_list)} valid images.")
    
    print("Writing metadata.jsonl...")
    with open(METADATA_FILE, "w") as f:
        for entry in metadata_list:
            f.write(json.dumps(entry) + "\n")
            
    print("Complete! Dataset is formatted and ready for the UNet.")

if __name__ == "__main__":
    asyncio.run(main())