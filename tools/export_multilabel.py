import xml.etree.ElementTree as ET
import csv
import json
from pathlib import Path
import os

XML_FILES = [
    "data/annotated/annotations_final.xml", # wide_shot
    "data/annotated/cvat_expression.xml"    # expression
]
OUTPUT_DIR = Path("data/annotated")

def map_attrs_to_yolo(label, attrs):
    if label == "empty_seat":
        return [7]
    
    ids = []
    if attrs.get("is_focused") == "true": ids.append(1)
    if attrs.get("is_drowsy") == "true": ids.append(2)
    if attrs.get("is_sleeping") == "true": ids.append(3)
    if attrs.get("is_using_phone") == "true": ids.append(4)
    if attrs.get("is_off_task") == "true": ids.append(5)
    if attrs.get("is_side_talking") == "true": ids.append(6)
    if attrs.get("is_raising_hand") == "true": ids.append(8)
    
    if not ids:
        ids.append(0) # generic person
        
    return ids

def main():
    all_data = [] # for JSON
    csv_rows = []
    
    # CSV Header
    csv_headers = ["image_path", "xmin", "ymin", "xmax", "ymax", "label", 
                   "is_focused", "is_drowsy", "is_sleeping", "is_using_phone", 
                   "is_off_task", "is_side_talking", "is_raising_hand"]
    csv_rows.append(csv_headers)
    
    for xml_file in XML_FILES:
        if not os.path.exists(xml_file):
            print(f"Skipping {xml_file}")
            continue
            
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for img in root.findall('image'):
            img_name = img.get('name')
            w = float(img.get('width'))
            h = float(img.get('height'))
            
            # Extract category from name (e.g., wide_shot/goc_cheo__frame_... or just goc_cheo...)
            # If the name does not have category, we infer it.
            category = "wide_shot" if "goc_" in img_name else "expression"
            if "/" in img_name:
                category, img_name = img_name.split("/")[-2:]
                
            txt_path = OUTPUT_DIR / category / "labels" / img_name.replace(".jpg", ".txt")
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            
            yolo_lines = []
            img_data = {
                "image": img_name,
                "category": category,
                "width": w,
                "height": h,
                "objects": []
            }
            
            for box in img.findall('box'):
                lbl = box.get('label')
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                attrs = {a.get('name'): a.text for a in box.findall('attribute')}
                yolo_ids = map_attrs_to_yolo(lbl, attrs)
                
                # Convert to YOLO
                cx = ((xtl + xbr) / 2) / w
                cy = ((ytl + ybr) / 2) / h
                bw = (xbr - xtl) / w
                bh = (ybr - ytl) / h
                
                for y_id in yolo_ids:
                    yolo_lines.append(f"{y_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
                # Add to JSON
                obj_data = {
                    "label": lbl,
                    "bbox": [xtl, ytl, xbr, ybr],
                    "attributes": attrs
                }
                img_data["objects"].append(obj_data)
                
                # Add to CSV
                row = [
                    f"{category}/{img_name}", xtl, ytl, xbr, ybr, lbl,
                    1 if attrs.get("is_focused")=="true" else 0,
                    1 if attrs.get("is_drowsy")=="true" else 0,
                    1 if attrs.get("is_sleeping")=="true" else 0,
                    1 if attrs.get("is_using_phone")=="true" else 0,
                    1 if attrs.get("is_off_task")=="true" else 0,
                    1 if attrs.get("is_side_talking")=="true" else 0,
                    1 if attrs.get("is_raising_hand")=="true" else 0
                ]
                csv_rows.append(row)
                
            # Write YOLO txt
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines) + "\n")
                
            all_data.append(img_data)
            
    # Write JSON
    json_path = OUTPUT_DIR / "dataset_multilabel.json"
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2)
        
    # Write CSV
    csv_path = OUTPUT_DIR / "dataset_multilabel.csv"
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
        
    print(f"Exported YOLO txts to {OUTPUT_DIR}/[category]/labels/")
    print(f"Exported JSON to {json_path}")
    print(f"Exported CSV to {csv_path}")

if __name__ == "__main__":
    main()
