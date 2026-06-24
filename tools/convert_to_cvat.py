import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import cv2

ANNOTATED_DIR = Path("data/annotated")

# Mapping from old YOLO ID to CVAT label + attribute
# Old IDs:
# 0: person (generic walking around)
# 1: focused
# 2: drowsy
# 3: sleeping
# 4: using_phone
# 5: off_task
# 6: side_talking
# 7: away_from_seat
# 8: raising_hand

ATTRIBUTES = [
    "is_focused", "is_drowsy", "is_sleeping", 
    "is_using_phone", "is_off_task", "is_side_talking", "is_raising_hand"
]

def map_old_id_to_cvat(old_id):
    if old_id == 7:
        return "empty_seat", None
    
    # Everything else is a 'person'
    attr = None
    if old_id == 1: attr = "is_focused"
    elif old_id == 2: attr = "is_drowsy"
    elif old_id == 3: attr = "is_sleeping"
    elif old_id == 4: attr = "is_using_phone"
    elif old_id == 5: attr = "is_off_task"
    elif old_id == 6: attr = "is_side_talking"
    elif old_id == 8: attr = "is_raising_hand"
    # If old_id == 0, attr is None (just a generic person walking)
    
    return "person", attr

def create_meta_node():
    meta = ET.Element("meta")
    task = ET.SubElement(meta, "task")
    labels = ET.SubElement(task, "labels")
    
    # Person Label
    person_lbl = ET.SubElement(labels, "label")
    ET.SubElement(person_lbl, "name").text = "person"
    attrs = ET.SubElement(person_lbl, "attributes")
    
    for attr_name in ATTRIBUTES:
        attr = ET.SubElement(attrs, "attribute")
        ET.SubElement(attr, "name").text = attr_name
        ET.SubElement(attr, "mutable").text = "True"
        ET.SubElement(attr, "input_type").text = "checkbox"
        ET.SubElement(attr, "default_value").text = "false"
        ET.SubElement(attr, "values").text = "false\ntrue"
        
    # Empty Seat Label
    seat_lbl = ET.SubElement(labels, "label")
    ET.SubElement(seat_lbl, "name").text = "empty_seat"
    ET.SubElement(seat_lbl, "attributes")
    
    return meta

def main():
    # Process both wide_shot and expression separately
    for category in ["wide_shot", "expression"]:
        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        root.append(create_meta_node())
        
        image_id = 0
        
        images_dir = ANNOTATED_DIR / category / "images"
        labels_dir = ANNOTATED_DIR / category / "labels"
        
        if not images_dir.exists(): continue
            
        for img_path in sorted(images_dir.glob("*.jpg")):
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            
            # Get Image Dimensions
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w, _ = img.shape
            
            img_node = ET.SubElement(root, "image", id=str(image_id), name=f"{img_path.name}", width=str(w), height=str(h))
            image_id += 1
            
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_id = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            
                            # Convert YOLO normalized to absolute
                            xtl = (cx - bw/2) * w
                            ytl = (cy - bh/2) * h
                            xbr = (cx + bw/2) * w
                            ybr = (cy + bh/2) * h
                            
                            label_name, true_attr = map_old_id_to_cvat(old_id)
                            
                            box_node = ET.SubElement(img_node, "box", label=label_name, occluded="0", 
                                                     xtl=f"{xtl:.2f}", ytl=f"{ytl:.2f}", xbr=f"{xbr:.2f}", ybr=f"{ybr:.2f}")
                            
                            if label_name == "person":
                                for attr_name in ATTRIBUTES:
                                    val = "true" if attr_name == true_attr else "false"
                                    ET.SubElement(box_node, "attribute", name=attr_name).text = val

        # Save XML for this category
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        out_path = ANNOTATED_DIR / f"cvat_{category}.xml"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(xmlstr)
            
        print(f"Successfully generated CVAT XML at {out_path}")
        print(f"Total images processed for {category}: {image_id}")

if __name__ == "__main__":
    main()
