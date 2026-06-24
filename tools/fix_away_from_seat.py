import argparse
import os
from pathlib import Path

LABELS_DIR = Path("data/annotated/wide_shot/labels")

def calculate_iou(box1, box2):
    """
    Calculate IoU of two bounding boxes in normalized (cx, cy, w, h) format.
    """
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2

    # Convert to x1, y1, x2, y2
    b1_x1, b1_y1 = cx1 - w1 / 2, cy1 - h1 / 2
    b1_x2, b1_y2 = cx1 + w1 / 2, cy1 + h1 / 2
    b2_x1, b2_y1 = cx2 - w2 / 2, cy2 - h2 / 2
    b2_x2, b2_y2 = cx2 + w2 / 2, cy2 + h2 / 2

    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = w1 * h1
    b2_area = w2 * h2

    return inter_area / (b1_area + b2_area - inter_area)

def fix_labels(prefix):
    # Find frame 1
    frame1_path = LABELS_DIR / f"{prefix}__frame_00001.txt"
    if not frame1_path.exists():
        print(f"[ERROR] Cannot find {frame1_path}")
        return

    # Extract anchor boxes
    anchor_boxes = []
    with open(frame1_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and int(parts[0]) == 7: # away_from_seat is 7
                cx, cy, w, h = map(float, parts[1:5])
                anchor_boxes.append((cx, cy, w, h))

    print(f"[{prefix}] Found {len(anchor_boxes)} 'away_from_seat' boxes in frame 1.")
    if len(anchor_boxes) == 0:
        return

    # Process all other frames
    all_txts = sorted([p for p in LABELS_DIR.glob(f"{prefix}*.txt") if p.name != frame1_path.name and "aug" not in p.name])
    
    fixed_count = 0
    for txt_path in all_txts:
        new_lines = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                
                # Check overlap with anchors
                is_anchor = False
                for anchor in anchor_boxes:
                    iou = calculate_iou((cx, cy, w, h), anchor)
                    if iou > 0.5:  # High overlap means it's the same chair
                        is_anchor = True
                        break
                
                if is_anchor and cls_id != 7:
                    cls_id = 7
                    fixed_count += 1
                
                new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                
        # Write back
        with open(txt_path, 'w') as f:
            f.write("\n".join(new_lines) + "\n")

    print(f"[{prefix}] Fixed {fixed_count} labels across {len(all_txts)} frames.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    fix_labels(args.prefix)

if __name__ == "__main__":
    main()
