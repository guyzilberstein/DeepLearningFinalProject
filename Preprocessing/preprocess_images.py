import os
import time
from PIL import Image
from pathlib import Path

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

def preprocess_images(source_folder, dest_folder, target_size=(256, 256)):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    source_path = Path(source_folder)
    
    # Get list of all source files recursively
    files = []
    for ext in ['*.HEIC', '*.heic', '*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        files.extend(list[Path](source_path.rglob(ext)))
        
    print(f"Scanning {len(files)} files in source (recursively)...")
    
    skipped_count = 0
    processed_count = 0
    start_time = time.time()
    
    for i, file_path in enumerate(files):
        # 1. Determine what the output filename WOULD be
        # We use just the filename, flattening the directory structure
        # BEWARE: This assumes unique filenames across subdirectories, which is true for iOS IMG_XXXX.HEIC
        new_filename = file_path.stem + ".jpg"
        save_path = os.path.join(dest_folder, new_filename)
        
        # 2. THE CHECK: Does this file already exist?
        if os.path.exists(save_path):
            skipped_count += 1
            continue  # Skip to the next file immediately
            
        # 3. If we are here, the file is new. Process it!
        try:
            image = Image.open(file_path)
            image = image.convert('RGB')
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            image.save(save_path, "JPEG", quality=90)
            
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} new images...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    end_time = time.time()
    print(f"Done! Processed {processed_count} new images.")
    print(f"Skipped {skipped_count} existing images.")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Default paths assuming script is run from project root or Preprocessing/ folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    source_dir = os.path.join(project_root, "Photos")
    dest_dir = os.path.join(project_root, "ProcessedImages")
    
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
    else:
        preprocess_images(source_dir, dest_dir)
