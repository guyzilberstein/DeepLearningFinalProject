def preprocess_images(source_folder, dest_folder, target_size=(256, 256)):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Get list of all source files
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.heic', '.jpg', '.jpeg', '.png'))]
    print(f"Scanning {len(files)} files in source...")
    
    skipped_count = 0
    processed_count = 0
    start_time = time.time()
    
    for i, filename in enumerate(files):
        # 1. Determine what the output filename WOULD be
        new_filename = os.path.splitext(filename)[0] + ".jpg"
        save_path = os.path.join(dest_folder, new_filename)
        
        # 2. THE CHECK: Does this file already exist?
        if os.path.exists(save_path):
            skipped_count += 1
            continue  # Skip to the next file immediately
            
        # 3. If we are here, the file is new. Process it!
        try:
            img_path = os.path.join(source_folder, filename)
            image = Image.open(img_path)
            image = image.convert('RGB')
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            image.save(save_path, "JPEG", quality=90)
            
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} new images...")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    end_time = time.time()
    print(f"Done! Processed {processed_count} new images.")
    print(f"Skipped {skipped_count} existing images.")
    print(f"Time taken: {end_time - start_time:.2f} seconds.")
