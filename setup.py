import os
import zipfile
import shutil

def main():
    print("Setting up directory structure...")
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("src", exist_ok=True)
    os.makedirs("report", exist_ok=True)
    os.makedirs("data/qdrant_storage", exist_ok=True)
    
    zip_path = "val_test2020.zip"
    dest_dir = "data/images"
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} not found.")
        return
        
    print(f"Unzipping a subset of images from {zip_path} to {dest_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Get all files inside the 'test' directory that are jpg images
        all_files = [f for f in z.namelist() if f.startswith('test/') and f.endswith('.jpg')]
        print(f"Total image files available in zip: {len(all_files)}")
        
        # Take the first 1000 images
        subset = all_files[:1000]
        print(f"Extracting {len(subset)} images...")
        
        extracted_count = 0
        for file_info in subset:
            # We want to extract it and save it flat under data/images/
            filename = os.path.basename(file_info)
            dest_file_path = os.path.join(dest_dir, filename)
            
            # Read from zip and write to destination
            with z.open(file_info) as source, open(dest_file_path, 'wb') as target:
                shutil.copyfileobj(source, target)
            extracted_count += 1
            if extracted_count % 100 == 0:
                print(f"Extracted {extracted_count}/1000...")
                
    print(f"Successfully extracted {extracted_count} images to {dest_dir}.")
    print("Setup completed successfully.")

if __name__ == "__main__":
    main()
