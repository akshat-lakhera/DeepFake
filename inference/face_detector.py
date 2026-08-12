import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
        
    def extract_face(self, image_rgb, target_size=(380, 380), margin=20):
        """
        Detects and extracts the largest face from an RGB image.
        Returns:
            face_img: The cropped and resized face as a numpy array, or None if no face is found.
            box: The bounding box of the face [x, y, w, h]
        """
        results = self.detector.detect_faces(image_rgb)
        
        if not results:
            return None, None
            
        # Get largest face by area
        largest_face = max(results, key=lambda b: b['box'][2] * b['box'][3])
        x, y, w, h = largest_face['box']
        
        # Add margin to capture full head
        img_h, img_w, _ = image_rgb.shape
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_w, x + w + margin)
        y2 = min(img_h, y + h + margin)
        
        face_crop = image_rgb[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None, None
            
        face_resized = cv2.resize(face_crop, target_size)
        return face_resized, [x1, y1, x2-x1, y2-y1]

    def extract_faces_from_video(self, video_path, frame_skip=10, max_frames=50):
        """
        Extracts faces from a video file.
        Returns:
            A list of cropped face images.
        """
        cap = cv2.VideoCapture(video_path)
        faces = []
        frame_idx = 0
        
        while cap.isOpened() and len(faces) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_skip == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_img, _ = self.extract_face(img_rgb)
                if face_img is not None:
                    faces.append(face_img)
                    
            frame_idx += 1
            
        cap.release()
        return faces
