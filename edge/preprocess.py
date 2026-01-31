import cv2
import numpy as np
from typing import Dict, Any

class ImagePreprocessPipeline:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def assess_dimensions(self, image: np.ndarray) -> Dict[str, Any]: 
        height, width, channels = image.shape
        aspect_ratio = width / height
        return {
            "width": width,
            "height": height,
            "channels": channels,
            "aspect_ratio": aspect_ratio,
            "pixel_count": width * height
        }

    def process(self, image: np.ndarray) -> np.ndarray: 
        processed_img = cv2.resize(image, self.target_size)
        processed_img = processed_img.astype(np.float32) / 255.0
        
        return processed_img

    def run(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return {"status": "error", "message": "Failed to decode image"}
 
            dimensions = self.assess_dimensions(img)
 
            processed_img = self.process(img)
 
            return {
                "status": "success",
                "original_dimensions": dimensions,
                "processed_shape": processed_img.shape,
                "message": "Image preprocessed successfully",
                "placeholder_processing": "Resized and normalized"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
 
pipeline = ImagePreprocessPipeline()

def run_pipeline(image_bytes: bytes):
    return pipeline.run(image_bytes)