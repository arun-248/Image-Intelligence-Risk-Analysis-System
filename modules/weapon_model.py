"""
download_weapon_model.py - Download and verify all detection models
Run once: python download_weapon_model.py
"""

import os
import sys

print("=" * 80)
print("VISIONIQ - WEAPON DETECTION MODEL SETUP")
print("=" * 80)
print("\nDownloading detection models (one-time setup, ~200MB total)...")
print("This may take 2-5 minutes depending on your internet speed.\n")

try:
    from ultralytics import YOLO
    
    models_to_download = [
        ("yolov8n.pt", "YOLOv8n-COCO", "80 common objects"),
        ("yolov8n-oiv7.pt", "YOLOv8n-OIV7", "600 objects including weapons"),
    ]
    
    downloaded = []
    
    for model_file, model_name, description in models_to_download:
        print(f"\n{'─' * 80}")
        print(f"Model: {model_name}")
        print(f"Description: {description}")
        print(f"File: {model_file}")
        
        if os.path.exists(model_file):
            print(f"✓ Already downloaded")
            downloaded.append(model_name)
        else:
            print(f"⬇ Downloading...")
            try:
                model = YOLO(model_file)
                print(f"✓ Download complete")
                downloaded.append(model_name)
            except Exception as e:
                print(f"✗ Download failed: {e}")
    
    print(f"\n{'=' * 80}")
    print(f"SETUP COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n✓ Downloaded {len(downloaded)}/{len(models_to_download)} models:")
    for name in downloaded:
        print(f"  • {name}")
    
    if len(downloaded) == len(models_to_download):
        print("\n✅ ALL MODELS READY!")
        print("\nYou can now run: streamlit run app.py")
    else:
        print("\n⚠ Some models failed to download")
        print("The system will still work but with limited detection")
    
    print(f"\n{'=' * 80}\n")

except ImportError:
    print("\n❌ ERROR: ultralytics package not found")
    print("\nInstall it with:")
    print("  pip install ultralytics --break-system-packages")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ UNEXPECTED ERROR: {e}")
    print("\nPlease report this error if the problem persists")
    sys.exit(1)