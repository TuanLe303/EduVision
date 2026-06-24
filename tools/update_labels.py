import os
import glob
import re

data_dir = r"C:\Users\an\anCloud\Projects\EduVision\data"

label_map = {
    '1': '0',
    '2': '1',
    '3': '2',
    '4': '3',
    '5': '4',
    '6': '5',
    '8': '6',
}

new_classes = [
    "focused",
    "drowsy",
    "sleeping",
    "using_phone",
    "off_task",
    "side_talking",
    "raising_hand"
]

def main():
    print("Starting label update...")
    
    # 1. Update dataset.yaml
    yaml_files = glob.glob(os.path.join(data_dir, '**', 'dataset.yaml'), recursive=True)
    for yaml_file in yaml_files:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace nc:
        content = re.sub(r'nc:\s*\d+', 'nc: 7', content)
        
        # Replace names: block
        names_block = "names:\n  0: focused\n  1: drowsy\n  2: sleeping\n  3: using_phone\n  4: off_task\n  5: side_talking\n  6: raising_hand"
        content = re.sub(r'names:\s*\n(?:\s+\d+:\s+\w+\n?)+', names_block + '\n', content)
        
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {yaml_file}")

    # 2. Update classes.txt
    classes_files = glob.glob(os.path.join(data_dir, '**', 'classes.txt'), recursive=True)
    for cfile in classes_files:
        with open(cfile, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_classes) + '\n')
        print(f"Updated {cfile}")

    # 3. Update labels
    txt_files = glob.glob(os.path.join(data_dir, '**', '*.txt'), recursive=True)
    txt_files = [f for f in txt_files if not f.endswith('classes.txt')]

    modified_count = 0
    scanned_count = 0
    for txt_file in txt_files:
        scanned_count += 1
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        changed = False
        is_yolo_format = True
        
        if not lines:
            continue
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5 or not parts[0].isdigit():
                is_yolo_format = False
                break
            
            # Check if coords are floats between 0 and 1
            try:
                for p in parts[1:5]:
                    val = float(p)
                    # Coordinates could technically be slightly out of bounds due to augmentation, 
                    # but typically they are between -0.1 and 1.1 max. We use a safe range.
                    if val < -0.5 or val > 1.5:
                        is_yolo_format = False
                        break
            except ValueError:
                is_yolo_format = False
                break
                
            if not is_yolo_format:
                break
                
        if not is_yolo_format:
            continue
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = parts[0]
            if class_id == '0' or class_id == '7':
                changed = True
                continue
            if class_id in label_map:
                parts[0] = label_map[class_id]
                new_lines.append(' '.join(parts))
                changed = True
            else:
                new_lines.append(line)
                
        if changed:
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines) + '\n' if new_lines else '')
            modified_count += 1

    print(f"Scanned {scanned_count} text files.")
    print(f"Successfully modified labels in {modified_count} files.")

if __name__ == '__main__':
    main()
