# HEIC to JPEG Converter

This script converts HEIC images to JPEG format while preserving the original files.

## Requirements

- Python 3.6 or higher
- Required Python packages (install using `pip install -r requirements.txt`):
  - Pillow
  - pillow-heif

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AlfaroLab/oist-field-collection-2025.git
cd oist-field-collection-2025
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the HEIC images in the `Hauls/haul_001` directory
2. Run the conversion script:
```bash
python convert.py
```

The script will:
- Walk through the directory structure
- Find all HEIC files
- Create a new `JPEGversion` directory
- Convert HEIC files to JPEG format while preserving the original files
- Maintain the same directory structure in the output

## Notes

- Original HEIC files are preserved
- JPEG files are saved with maximum quality (100)
- The script handles different color spaces and transparency
- Error messages will be displayed if any conversion fails 