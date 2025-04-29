import os
from pathlib import Path
from pillow_heif import register_heif_opener
from PIL import Image

def convert_heic_to_jpeg(input_dir, output_dir):
    """
    Convert all HEIC images in the input directory to JPEG format.
    
    Args:
        input_dir (str): Directory containing HEIC images
        output_dir (str): Directory to save JPEG images
    """
    # Register HEIF opener
    register_heif_opener()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Walk through the directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.heic'):
                # Get full path of input file
                input_path = os.path.join(root, file)
                
                # Create corresponding output path
                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Create output filename (same name but with .jpg extension)
                output_filename = os.path.splitext(file)[0] + '.jpg'
                output_path = os.path.join(output_subdir, output_filename)
                
                try:
                    # Open and convert the image
                    with Image.open(input_path) as img:
                        # Convert to RGB if necessary (HEIC might be in different color space)
                        if img.mode in ('RGBA', 'LA'):
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[-1])
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save as JPEG with maximum quality
                        img.save(output_path, 'JPEG', quality=100)
                        print(f"Converted: {input_path} -> {output_path}")
                
                except Exception as e:
                    print(f"Error converting {input_path}: {str(e)}")

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output directories
    input_dir = os.path.join(current_dir, "Hauls", "haul_001")
    output_dir = os.path.join(current_dir, "JPEGversion")
    
    # Convert images
    convert_heic_to_jpeg(input_dir, output_dir)
    print("Conversion complete!") 