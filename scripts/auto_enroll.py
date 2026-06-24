import cv2
import numpy as np
import json
import os
import random
from insightface.app import FaceAnalysis
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

def main():
    video_path = r"data\test\test4\test4.mp4"
    output_images_dir = r"data\face\auto_enrollment_images4"
    output_json = r"data\face\auto_enrollments4.json"
    
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Determine providers
    try:
        import onnxruntime
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else ["CPUExecutionProvider"]
    except ImportError:
        providers = ["CPUExecutionProvider"]

    print(f"Using providers: {providers}")

    # Initialize face analysis
    app = FaceAnalysis(name="buffalo_s", providers=providers)
    app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(640, 640))
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30
        
    frame_interval = 5 # extract face every 5 frames
    
    embeddings = []
    face_images = []
    
    print("Reading video and extracting faces...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            faces = app.get(frame)
            for face in faces:
                emb = getattr(face, "normed_embedding", None)
                if emb is None:
                    emb = getattr(face, "embedding", None)
                if emb is not None:
                    # Normalize
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
                    
                    if x2 > x1 and y2 > y1:
                        face_crop = frame[y1:y2, x1:x2].copy()
                        embeddings.append(emb)
                        face_images.append(face_crop)
        
        frame_idx += 1
        
    cap.release()
    
    print(f"Total faces extracted: {len(embeddings)}")
    if len(embeddings) == 0:
        print("No faces found!")
        return

    # Clustering
    print("Clustering faces...")
    emb_array = np.array(embeddings)
    # Cosine distance linkage
    Z = linkage(emb_array, method='average', metric='cosine')
    
    # Distance threshold 0.4 means similarity > 0.6
    clusters = fcluster(Z, t=0.4, criterion='distance')
    
    # Count cluster sizes
    cluster_counts = defaultdict(int)
    for c in clusters:
        cluster_counts[c] += 1
        
    # Get top 16 clusters
    top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:16]
    
    print(f"Found {len(top_clusters)} major clusters (top 16 will be used).")
    
    enrollments = {
        "metadata": {
            "version": 1,
            "model": "buffalo_s",
            "embedding_dimension": 512
        },
        "students": []
    }
    
    for i, (cluster_id, count) in enumerate(top_clusters):
        student_id = f"Student_{i+1:02d}"
        
        # Find all indices for this cluster
        indices = [idx for idx, c in enumerate(clusters) if c == cluster_id]
        
        # Sample up to 5
        sampled_indices = random.sample(indices, min(5, len(indices)))
        
        student_dir = os.path.join(output_images_dir, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        student_embeddings = []
        for j, idx in enumerate(sampled_indices):
            emb = embeddings[idx]
            face_img = face_images[idx]
            
            # Save image
            img_path = os.path.join(student_dir, f"face_{j+1}.jpg")
            cv2.imwrite(img_path, face_img)
            
            student_embeddings.append(emb.tolist())
            
        enrollments["students"].append({
            "student_id": student_id,
            "name": student_id,
            "embeddings": student_embeddings
        })
        
        print(f"Created {student_id} from cluster {cluster_id} with {len(student_embeddings)} embeddings (total extracted size: {count})")
        
    with open(output_json, 'w') as f:
        json.dump(enrollments, f, indent=2)
        
    print(f"Saved {output_json}")

if __name__ == '__main__':
    main()
