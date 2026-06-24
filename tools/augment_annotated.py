import os
from pathlib import Path
import cv2
import albumentations as A
from tqdm import tqdm

ANNOTATED_DIR = Path("data/annotated")

def build_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.Perspective(scale=(0.02, 0.06), p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.6),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_folder(category, pipeline, multiplier=2):
    images_dir = ANNOTATED_DIR / category / "images"
    labels_dir = ANNOTATED_DIR / category / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"Skipping {category}, directories not found.")
        return

    # Find original images
    originals = [f for f in images_dir.glob("*.jpg") if "_aug" not in f.stem]
    
    print(f"Augmenting {category}: {len(originals)} original images found.")
    
    for img_path in tqdm(originals, desc=f"Augmenting {category}"):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        
        if not lbl_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Read YOLO labels
        bboxes = []
        class_labels = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # Pre-clip to avoid Albumentations ValueError
                    cx = max(0.000001, min(0.999999, cx))
                    cy = max(0.000001, min(0.999999, cy))
                    w = max(0.000001, min(0.999999, w))
                    h = max(0.000001, min(0.999999, h))
                    
                    # Also ensure x_min, y_min, x_max, y_max are strictly inside [0, 1]
                    x_min = cx - w / 2
                    y_min = cy - h / 2
                    x_max = cx + w / 2
                    y_max = cy + h / 2
                    
                    if x_min < 0:
                        cx -= x_min
                        w += x_min * 2  # shrink width to fit
                    if y_min < 0:
                        cy -= y_min
                        h += y_min * 2
                    if x_max > 1:
                        cx -= (x_max - 1)
                        w -= (x_max - 1) * 2
                    if y_max > 1:
                        cy -= (y_max - 1)
                        h -= (y_max - 1) * 2
                        
                    # Re-clip just in case of floating point errors
                    cx = max(0.000001, min(0.999999, cx))
                    cy = max(0.000001, min(0.999999, cy))
                    w = max(0.000001, min(0.999999, w))
                    h = max(0.000001, min(0.999999, h))

                    bboxes.append([cx, cy, w, h])
                    class_labels.append(cls_id)
        
        for i in range(1, multiplier + 1):
            aug_img_name = f"{img_path.stem}_aug{i}.jpg"
            aug_lbl_name = f"{img_path.stem}_aug{i}.txt"
            aug_img_path = images_dir / aug_img_name
            aug_lbl_path = labels_dir / aug_lbl_name
            
            # Apply augmentation
            augmented = pipeline(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
            
            # Save Image
            aug_bgr = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_img_path), aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Save Labels
            with open(aug_lbl_path, "w") as f:
                for bbox, cls_id in zip(augmented["bboxes"], augmented["class_labels"]):
                    # Clip bboxes just in case
                    cx, cy, w, h = bbox
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w = max(0.001, min(1.0, w))
                    h = max(0.001, min(1.0, h))
                    f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def main():
    pipeline = build_pipeline()
    augment_folder("wide_shot", pipeline, multiplier=2)
    augment_folder("expression", pipeline, multiplier=2)
    print("Augmentation complete!")

if __name__ == "__main__":
    main()
