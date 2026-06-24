import argparse
import base64
import cv2
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai not installed. Run: pip install openai")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "data" / "annotated" / "wide_shot" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "annotated" / "wide_shot" / "labels"

BEHAVIOR_CLASSES = [
    "person",          # 0
    "focused",         # 1
    "drowsy",          # 2
    "sleeping",        # 3
    "using_phone",     # 4
    "off_task",        # 5
    "side_talking",    # 6
    "away_from_seat",  # 7
    "raising_hand",    # 8
]
BEHAVIOR_CLASS_ID = {name: i for i, name in enumerate(BEHAVIOR_CLASSES)}

def yolo_to_xywh(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    # Convert to pixel coordinates
    abs_w = int(w * img_w)
    abs_h = int(h * img_h)
    abs_x = int((cx - w/2) * img_w)
    abs_y = int((cy - h/2) * img_h)
    
    # Ensure they are within image bounds
    abs_x = max(0, abs_x)
    abs_y = max(0, abs_y)
    return cls_id, abs_x, abs_y, abs_w, abs_h

def xywh_to_yolo(cls_id, x, y, w, h, img_w, img_h):
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.001, min(1.0, nw))
    nh = max(0.001, min(1.0, nh))
    
    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

def get_gpt_label(client, image_crop):
    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', image_crop)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    prompt_text = (
        "You are an expert AI assisting in labeling student behavior. "
        "Look at this cropped image of a student in a classroom. "
        "Carefully analyze their body language and interactions to classify their behavior into exactly ONE of the following classes: "
        "focused, drowsy, sleeping, using_phone, off_task, side_talking, away_from_seat, raising_hand. "
        "Be confident in your classification. For example, if two students are facing each other or interacting, choose 'side_talking'. "
        "Only choose 'off_task' if they are clearly doing something unrelated to studying AND it does not fit any other specific category. "
        "Reply ONLY with the exact class name. No other text."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.0
        )
        prediction = response.choices[0].message.content.strip().lower()
        # Clean up any punctuation
        prediction = prediction.replace(".", "").replace(",", "").replace('"', '')
        
        if prediction in BEHAVIOR_CLASS_ID:
            return BEHAVIOR_CLASS_ID[prediction]
        else:
            return BEHAVIOR_CLASS_ID["off_task"] # Fallback
    except Exception as e:
        print(f"\n[ERROR] GPT API failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Track bounding boxes and label with GPT-4o.")
    parser.add_argument("--prefix", required=True, help="Prefix (e.g., goc_cheo)")
    parser.add_argument("--start", type=int, required=True, help="Start frame (must have a valid .txt label)")
    parser.add_argument("--end", type=int, required=True, help="End frame")
    parser.add_argument("--api-key", required=True, help="OpenAI API Key")
    parser.add_argument("--interval", type=int, default=2, help="Call GPT every N frames")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    # 1. Initialize from start frame
    start_img_name = f"{args.prefix}__frame_{args.start:05d}.jpg"
    start_lbl_name = f"{args.prefix}__frame_{args.start:05d}.txt"
    start_img_path = IMAGES_DIR / start_img_name
    start_lbl_path = LABELS_DIR / start_lbl_name

    if not start_img_path.exists() or not start_lbl_path.exists():
        print(f"[ERROR] Start frame image or label not found: {start_img_name}")
        sys.exit(1)

    frame1 = cv2.imread(str(start_img_path))
    if frame1 is None:
        print("[ERROR] Could not read start frame image.")
        sys.exit(1)
    
    img_h, img_w = frame1.shape[:2]

    # Read initial boxes
    trackers = []
    current_classes = []
    with open(start_lbl_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            res = yolo_to_xywh(line, img_w, img_h)
            if res:
                cls_id, x, y, w, h = res
                # Create MIL tracker for this box
                tracker = cv2.TrackerMIL_create()
                tracker.init(frame1, (x, y, w, h))
                trackers.append(tracker)
                current_classes.append(cls_id)
    
    print(f"Initialized {len(trackers)} trackers from frame {args.start}.")

    # 2. Process subsequent frames
    api_calls = 0
    for i in tqdm(range(args.start + 1, args.end + 1), desc="Tracking & Labeling"):
        img_name = f"{args.prefix}__frame_{i:05d}.jpg"
        lbl_name = f"{args.prefix}__frame_{i:05d}.txt"
        img_path = IMAGES_DIR / img_name
        lbl_path = LABELS_DIR / lbl_name
        
        if not img_path.exists():
            continue
            
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        new_labels = []
        call_gpt = ((i - args.start) % args.interval == 0)
        
        for t_idx, tracker in enumerate(trackers):
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                
                # GPT Labeling
                if call_gpt:
                    # Crop image for GPT
                    crop_x1 = max(0, x)
                    crop_y1 = max(0, y)
                    crop_x2 = min(img_w, x + w)
                    crop_y2 = min(img_h, y + h)
                    
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        new_cls_id = get_gpt_label(client, crop)
                        if new_cls_id is not None:
                            current_classes[t_idx] = new_cls_id
                            api_calls += 1
                
                cls_id = current_classes[t_idx]
                new_labels.append(xywh_to_yolo(cls_id, x, y, w, h, img_w, img_h))
            else:
                # Tracking failed for this object, keep old label (assume it didn't move much)
                # But without coords, we just skip it or log it. Let's skip to avoid bad coords.
                pass
                
        # Write new label file
        with open(lbl_path, "w") as f:
            f.write("\n".join(new_labels))
            
    print(f"\n[DONE] Processed up to frame {args.end}.")
    print(f"Total API calls made: {api_calls}")

if __name__ == "__main__":
    main()
