import xml.etree.ElementTree as ET
import os
import re
from pathlib import Path
import copy

ANNOTATIONS_XML = "data/annotated/annotations.xml"
YOLO_LABELS_DIR = Path("data/annotated/wide_shot/labels")

GOC_CHEO_KEYFRAMES = [1, 11, 14, 18, 32, 57, 89, 137, 139, 154, 173]

ATTRIBUTES = [
    "is_focused", "is_drowsy", "is_sleeping", 
    "is_using_phone", "is_off_task", "is_side_talking", "is_raising_hand"
]

def map_old_id_to_attr(old_id):
    if old_id == 7: return "empty_seat", None
    attr = None
    if old_id == 1: attr = "is_focused"
    elif old_id == 2: attr = "is_drowsy"
    elif old_id == 3: attr = "is_sleeping"
    elif old_id == 4: attr = "is_using_phone"
    elif old_id == 5: attr = "is_off_task"
    elif old_id == 6: attr = "is_side_talking"
    elif old_id == 8: attr = "is_raising_hand"
    return "person", attr

def calculate_iou(box1, box2):
    # box: [xmin, ymin, xmax, ymax]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def get_yolo_boxes(txt_path, img_w, img_h):
    # returns list of (cls_id, [xmin, ymin, xmax, ymax])
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                xmin = (cx - bw / 2) * img_w
                ymin = (cy - bh / 2) * img_h
                xmax = (cx + bw / 2) * img_w
                ymax = (cy + bh / 2) * img_h
                boxes.append((cls_id, [xmin, ymin, xmax, ymax]))
    return boxes

def process_goc_cheo(root):
    current_keyframe_boxes = []
    current_keyframe_idx = -1
    
    # Sort images by name just in case
    images = root.findall('image')
    goc_cheo_images = [img for img in images if "goc_cheo" in img.get('name')]
    
    # Extract frame number and sort
    def get_frame_num(name):
        m = re.search(r'frame_(\d+)', name)
        return int(m.group(1)) if m else 0
    
    goc_cheo_images.sort(key=lambda x: get_frame_num(x.get('name')))
    
    for img in goc_cheo_images:
        frame_num = get_frame_num(img.get('name'))
        
        if frame_num in GOC_CHEO_KEYFRAMES:
            # Save these boxes
            current_keyframe_boxes = [copy.deepcopy(box) for box in img.findall('box')]
            current_keyframe_idx = frame_num
        else:
            if current_keyframe_idx != -1:
                # Remove all existing boxes
                for box in img.findall('box'):
                    img.remove(box)
                # Append copies of keyframe boxes
                for box in current_keyframe_boxes:
                    img.append(copy.deepcopy(box))

def process_goc_thang_phai(root):
    images = root.findall('image')
    goc_phai_images = [img for img in images if "goc_thang_phai" in img.get('name')]
    
    for img in goc_phai_images:
        img_w = float(img.get('width'))
        img_h = float(img.get('height'))
        img_name = img.get('name')
        txt_path = YOLO_LABELS_DIR / img_name.replace(".jpg", ".txt")
        yolo_boxes = get_yolo_boxes(txt_path, img_w, img_h)
        
        for box_node in img.findall('box'):
            # Check if this box has attributes
            # A completely missing label box might have NO attributes, or all false.
            attrs = box_node.findall('attribute')
            has_true_attr = any(attr.text == 'true' for attr in attrs)
            is_person = box_node.get('label') == 'person'
            
            # If it's a person but no true attribute, OR if it has NO attributes at all
            if is_person and (not attrs or not has_true_attr):
                # Need to restore from YOLO txt
                xtl = float(box_node.get('xtl'))
                ytl = float(box_node.get('ytl'))
                xbr = float(box_node.get('xbr'))
                ybr = float(box_node.get('ybr'))
                cvat_bbox = [xtl, ytl, xbr, ybr]
                
                best_iou = 0
                best_cls = -1
                for yolo_cls, y_bbox in yolo_boxes:
                    iou = calculate_iou(cvat_bbox, y_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_cls = yolo_cls
                        
                if best_iou > 0.3:
                    lbl, attr_name = map_old_id_to_attr(best_cls)
                    box_node.set('label', lbl)
                    
                    # Remove existing attributes
                    for a in attrs:
                        box_node.remove(a)
                        
                    if lbl == 'person':
                        for a_name in ATTRIBUTES:
                            val = 'true' if a_name == attr_name else 'false'
                            new_attr = ET.SubElement(box_node, 'attribute', name=a_name)
                            new_attr.text = val

def main():
    tree = ET.parse(ANNOTATIONS_XML)
    root = tree.getroot()
    
    print("Processing goc_cheo forward-fill...")
    process_goc_cheo(root)
    
    print("Processing goc_thang_phai label restoration...")
    process_goc_thang_phai(root)
    
    out_path = "data/annotated/annotations_processed.xml"
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved processed XML to {out_path}")

if __name__ == "__main__":
    main()
