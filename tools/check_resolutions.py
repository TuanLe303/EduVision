import os
from pathlib import Path
from PIL import Image

def main():
    crops_dir = Path(r"c:\Users\an\anCloud\Projects\EduVision\data\special_crops")
    
    min_area = float('inf')
    max_area = 0
    
    min_res = None
    max_res = None
    
    min_path = ""
    max_path = ""
    
    count = 0
    
    for img_path in crops_dir.rglob("*.jpg"):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                area = w * h
                
                if area < min_area:
                    min_area = area
                    min_res = (w, h)
                    min_path = img_path
                    
                if area > max_area:
                    max_area = area
                    max_res = (w, h)
                    max_path = img_path
                    
                count += 1
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            
    print(f"Total images checked: {count}")
    if count > 0:
        print(f"Lowest resolution: {min_res[0]}x{min_res[1]} (Area: {min_area}) - File: {min_path.relative_to(crops_dir)}")
        print(f"Highest resolution: {max_res[0]}x{max_res[1]} (Area: {max_area}) - File: {max_path.relative_to(crops_dir)}")
    else:
        print("No images found.")

if __name__ == "__main__":
    main()
