import xml.etree.ElementTree as ET

tree = ET.parse("data/annotated/annotations.xml")
root = tree.getroot()

print("Root tags:", [child.tag for child in root])
images = root.findall('image')
tracks = root.findall('track')
print(f"Number of images: {len(images)}")
print(f"Number of tracks: {len(tracks)}")

if len(images) > 0:
    first_image = images[0]
    print(f"First image name: {first_image.get('name')}")
    boxes = first_image.findall('box')
    print(f"Boxes in first image: {len(boxes)}")
    if len(boxes) > 0:
        print(f"First box label: {boxes[0].get('label')}")
        attrs = boxes[0].findall('attribute')
        print(f"Attributes in first box: {len(attrs)}")
