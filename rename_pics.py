import os
import shutil
from pathlib import Path

def sequential_rename_images(source_folder, start_n=1, dest_folder_name="renamedpics"):
    """
    Copy images from source_folder to a new folder with sequential naming pic0001, pic0002, etc.
    
    Args:
        source_folder: Path to folder containing original images
        start_n: Starting number for sequence (e.g., 1 for pic0001)
        dest_folder_name: Name of the new folder to create (default: "renamedpics")
    """
    
    # Common image formats (includes raw formats)
    image_extensions = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
        '.webp', '.svg', '.raw', '.cr2', '.nef', '.arw', '.dng',
        '.heic', '.heif', '.ico', '.pcx', '.ppm', '.pgm', '.pbm'
    }
    
    source_path = Path(source_folder).resolve()
    
    # Check if source exists
    if not source_path.exists():
        print(f"❌ Error: Source folder '{source_folder}' does not exist!")
        return False
    
    if not source_path.is_dir():
        print(f"❌ Error: '{source_folder}' is not a directory!")
        return False
    
    # Create destination folder inside source folder (or change logic to create elsewhere)
    dest_path = source_path / dest_folder_name
    
    try:
        dest_path.mkdir(exist_ok=True)
        print(f"📁 Destination folder: {dest_path}")
    except Exception as e:
        print(f"❌ Error creating destination folder: {e}")
        return False
    
    # Get all image files (case-insensitive extension check)
    image_files = [
        f for f in source_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    # Sort for consistent ordering (optional: remove if you want random order)
    image_files.sort()
    
    if not image_files:
        print("⚠️  No image files found in the source folder.")
        return False
    
    print(f"🖼️  Found {len(image_files)} images")
    print(f"🔢 Starting sequence from: pic{start_n:04d}")
    print("-" * 50)
    
    success_count = 0
    skipped_count = 0
    
    for index, source_file in enumerate(image_files, start=start_n):
        # Generate new filename: pic0001.jpg, pic0002.png, etc.
        # Using :04d for 4-digit zero padding (0001, 0002... 9999)
        new_filename = f"pic{index:04d}{source_file.suffix.lower()}"
        dest_file = dest_path / new_filename
        
        # Handle case where file already exists in destination
        if dest_file.exists():
            print(f"⚠️  Skipping {source_file.name} -> {new_filename} (already exists)")
            skipped_count += 1
            continue
        
        try:
            # Copy file (preserves original)
            shutil.copy2(source_file, dest_file)
            print(f"✅ {source_file.name} -> {new_filename}")
            success_count += 1
        except Exception as e:
            print(f"❌ Error copying {source_file.name}: {e}")
    
    print("-" * 50)
    print(f"✨ Done! Successfully copied {success_count} files to '{dest_folder_name}/'")
    if skipped_count > 0:
        print(f"⚠️  Skipped {skipped_count} files (duplicates)")
    
    return True

def main():
    print("🖼️  Sequential Image Renamer")
    print("=" * 50)
    
    # Get user inputs
    source = input("📂 Enter source folder path: ").strip().strip('"')
    
    # Validate path
    while not os.path.exists(source):
        print("❌ Path not found!")
        source = input("📂 Enter valid source folder path: ").strip().strip('"')
    
    # Get starting number
    while True:
        try:
            start_num = input("🔢 Enter starting number (n) [default: 1]: ").strip()
            if not start_num:
                start_num = 1
                break
            start_num = int(start_num)
            if start_num < 0:
                print("❌ Please enter a positive number")
                continue
            break
        except ValueError:
            print("❌ Invalid number!")
    
    # Optional: custom destination folder name
    use_default = input("📁 Use default folder name 'renamedpics'? (y/n) [y]: ").strip().lower()
    if use_default == 'n':
        dest_name = input("Enter folder name: ").strip()
    else:
        dest_name = "renamedpics"
    
    print("\n" + "=" * 50)
    
    # Execute
    sequential_rename_images(source, start_num, dest_name)
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
