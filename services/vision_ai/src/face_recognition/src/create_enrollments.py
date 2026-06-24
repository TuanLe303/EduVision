"""
Script tạo file enrollment từ thư mục ảnh sinh viên.

Cấu trúc thư mục input mong đợi:
dataset_folder/
    student0/
        image1.jpg
        image2.jpg
    student1/
        image0.png

Cách chạy từ thư mục gốc của dự án (EduVision/):
ev\\Scripts\\python -m services.vision_ai.src.face_recognition.src.create_enrollments --input path/to/dataset --output data/enrollments.json --flip
"""
import argparse
from pathlib import Path
import cv2
import sys

from services.vision_ai.src.face_detection.src.face_detector import FaceDetector
from services.vision_ai.src.face_recognition.src.recognizer import FaceRecognizer

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create face enrollments from a directory of images.")
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to the input directory containing subdirectories for each student."
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path to save the output enrollments JSON file (e.g., data/enrollments.json)."
    )
    parser.add_argument(
        "--flip", 
        action="store_true", 
        help="Enable horizontal flip augmentation to enroll a pseudo-opposite profile."
    )
    return parser

def bbox_area(bbox) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))

def main():
    args = build_parser().parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory {input_dir} does not exist or is not a directory.")
        sys.exit(1)

    print("Initializing models...")
    # Khởi tạo detector và recognizer
    # Có thể tự động lấy device 'cuda' nếu có dựa trên config mặc định của project
    detector = FaceDetector(backend="scrfd")
    recognizer = FaceRecognizer(backend="insightface")

    total_enrolled = 0

    # Duyệt qua các thư mục con (mỗi thư mục là 1 sinh viên)
    for student_dir in input_dir.iterdir():
        if not student_dir.is_dir():
            continue
            
        student_id = student_dir.name
        print(f"\nProcessing student: {student_id}")

        for img_path in student_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [!] Failed to read image: {img_path.name}")
                continue

            # Phát hiện khuôn mặt
            faces = detector.detect(img)
            if not faces:
                print(f"  [!] No face detected in {img_path.name}")
                continue

            # Lấy khuôn mặt to nhất trong ảnh (tránh nhận nhầm người qua đường)
            best_face = max(faces, key=lambda f: bbox_area(f.bbox))

            try:
                # Tiến hành enroll
                recognizer.enroll(
                    student_id=student_id,
                    face_image=img,
                    name=student_id, # Tạm dùng student_id làm name, có thể sửa nếu cần
                    landmarks=best_face.landmarks
                )
                print(f"  [+] Enrolled from {img_path.name}")
                total_enrolled += 1
            except Exception as e:
                print(f"  [-] Failed to enroll from {img_path.name}: {e}")

            # Data Augmentation: Lật ngang ảnh (nếu có cờ --flip)
            if args.flip:
                flipped_img = cv2.flip(img, 1)
                flipped_faces = detector.detect(flipped_img)
                if flipped_faces:
                    best_flipped = max(flipped_faces, key=lambda f: bbox_area(f.bbox))
                    try:
                        recognizer.enroll(
                            student_id=student_id,
                            face_image=flipped_img,
                            name=student_id,
                            landmarks=best_flipped.landmarks
                        )
                        print(f"  [+] Enrolled from {img_path.name} (flipped)")
                        total_enrolled += 1
                    except Exception as e:
                        print(f"  [-] Failed to enroll flipped {img_path.name}: {e}")

    # Lưu kết quả xuống file JSON
    if total_enrolled > 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recognizer.save_enrollments(output_path)
        print(f"\nSuccessfully saved {total_enrolled} templates to {output_path}")
    else:
        print("\nNo templates were successfully enrolled. Output file was not created.")

if __name__ == "__main__":
    main()
