import xml.etree.ElementTree as ET
import cv2
import base64
import requests
import json
import numpy as np
from pathlib import Path
import math
import copy

ANNOTATIONS_XML = "data/annotated/annotations_processed.xml"
IMAGES_DIR = Path("data/raw_frames/wide_shot")
API_KEY = "YOUR_OPENAI_API_KEY"

ATTRIBUTES = [
    "is_focused", "is_drowsy", "is_sleeping", 
    "is_using_phone", "is_off_task", "is_side_talking", "is_raising_hand"
]

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top: return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)

def pad_and_resize(img, size=256):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    pad_img = np.zeros((size, size, 3), dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    pad_img[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return pad_img

def create_grid_image(crops):
    n = len(crops)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    size = 256
    grid = np.zeros((rows * size, cols * size, 3), dtype=np.uint8)
    
    for idx, crop in enumerate(crops):
        r = idx // cols
        c = idx % cols
        
        # Add index text
        padded = pad_and_resize(crop, size)
        cv2.rectangle(padded, (0, 0), (size-1, size-1), (0, 0, 255), 2)
        cv2.putText(padded, str(idx), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        
        grid[r*size:(r+1)*size, c*size:(c+1)*size] = padded
        
    return grid

def analyze_grid_with_gpt(grid_img, num_crops):
    _, buffer = cv2.imencode('.jpg', grid_img)
    b64_img = base64.b64encode(buffer).decode('utf-8')
    
    prompt = f"""
    Attached is a grid of {num_crops} cropped images of students in a classroom.
    Each crop has a green number in the top-left corner (from 0 to {num_crops-1}).
    Analyze their body language and return a valid JSON object. 
    The keys must be the string version of the numbers (e.g. "0", "1", "2").
    The values must be an array of applicable behaviors from this exact list:
    ["focused", "drowsy", "sleeping", "using_phone", "off_task", "side_talking", "raising_hand"].
    
    Rules:
    - If focused on the screen/lesson, return ["focused"]
    - If using phone, return ["using_phone", "off_task"]
    - If talking, return ["side_talking", "off_task"]
    - If drowsy or sleeping, return ["drowsy"] or ["sleeping"]
    - If ambiguous, you may assign multiple.
    
    Return ONLY a valid JSON object, no markdown, no other text.
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.2
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        # Clean markdown
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"GPT Error: {e}")
        return None

def process_target_set(root, prefix):
    images = root.findall('image')
    target_images = [img for img in images if prefix in img.get('name')]
    
    # Sort by frame
    def get_frame(name):
        import re
        m = re.search(r'frame_(\d+)', name)
        return int(m.group(1)) if m else 0
    target_images.sort(key=lambda x: get_frame(x.get('name')))
    
    previous_frame_boxes = [] # List of dict: {bbox: [x,y,x,y], attrs: {name: val}}
    
    for idx, img_node in enumerate(target_images):
        img_name = img_node.get('name')
        img_path = IMAGES_DIR / img_name
        
        boxes = [b for b in img_node.findall('box') if b.get('label') == 'person']
        
        if idx % 20 == 0:
            # KEYFRAME: Call GPT
            print(f"Keyframe [{prefix}]: {img_name} ({len(boxes)} persons)")
            if len(boxes) == 0:
                previous_frame_boxes = []
                continue
                
            frame_img = cv2.imread(str(img_path))
            if frame_img is None: continue
            
            crops = []
            valid_boxes = []
            for b in boxes:
                xtl, ytl, xbr, ybr = map(float, [b.get('xtl'), b.get('ytl'), b.get('xbr'), b.get('ybr')])
                x1, y1, x2, y2 = int(xtl), int(ytl), int(xbr), int(ybr)
                # Ensure within image bounds
                h, w = frame_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crops.append(frame_img[y1:y2, x1:x2])
                    valid_boxes.append(b)
            
            if not crops:
                previous_frame_boxes = []
                continue
                
            grid = create_grid_image(crops)
            gpt_results = analyze_grid_with_gpt(grid, len(crops))
            
            if gpt_results:
                previous_frame_boxes = []
                for i, b in enumerate(valid_boxes):
                    labels = gpt_results.get(str(i), ["focused"])
                    
                    # Map to CVAT attributes
                    attr_map = {attr: "false" for attr in ATTRIBUTES}
                    for lbl in labels:
                        if lbl == "focused": attr_map["is_focused"] = "true"
                        elif lbl == "drowsy": attr_map["is_drowsy"] = "true"
                        elif lbl == "sleeping": attr_map["is_sleeping"] = "true"
                        elif lbl == "using_phone": attr_map["is_using_phone"] = "true"
                        elif lbl == "off_task": attr_map["is_off_task"] = "true"
                        elif lbl == "side_talking": attr_map["is_side_talking"] = "true"
                        elif lbl == "raising_hand": attr_map["is_raising_hand"] = "true"
                    
                    # Update XML
                    for a in b.findall('attribute'): b.remove(a)
                    for k, v in attr_map.items():
                        new_attr = ET.SubElement(b, 'attribute', name=k)
                        new_attr.text = v
                    
                    # Save for next frames
                    xtl, ytl, xbr, ybr = map(float, [b.get('xtl'), b.get('ytl'), b.get('xbr'), b.get('ybr')])
                    previous_frame_boxes.append({
                        "bbox": [xtl, ytl, xbr, ybr],
                        "attrs": attr_map
                    })
        else:
            # INTERMEDIATE FRAME: Forward Fill via IoU
            new_previous_boxes = []
            for b in boxes:
                xtl, ytl, xbr, ybr = map(float, [b.get('xtl'), b.get('ytl'), b.get('xbr'), b.get('ybr')])
                curr_bbox = [xtl, ytl, xbr, ybr]
                
                best_iou = 0
                best_attrs = None
                for prev_b in previous_frame_boxes:
                    iou = calculate_iou(curr_bbox, prev_b['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_attrs = prev_b['attrs']
                
                if best_iou > 0.3 and best_attrs:
                    # Apply tracked attributes
                    for a in b.findall('attribute'): b.remove(a)
                    for k, v in best_attrs.items():
                        new_attr = ET.SubElement(b, 'attribute', name=k)
                        new_attr.text = v
                        
                    # Update tracking bbox for next frame
                    new_previous_boxes.append({
                        "bbox": curr_bbox,
                        "attrs": best_attrs
                    })
            previous_frame_boxes = new_previous_boxes

def main():
    tree = ET.parse(ANNOTATIONS_XML)
    root = tree.getroot()
    
    print("Processing goc_thang_phai...")
    process_target_set(root, "goc_thang_phai")
    
    print("Processing goc_thang_trai...")
    process_target_set(root, "goc_thang_trai")
    
    out_path = "data/annotated/annotations_final.xml"
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved final XML to {out_path}")

if __name__ == "__main__":
    main()
