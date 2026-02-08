import unittest
import numpy as np
import cv2
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge.jax_train import run_inference

class TestJaxInference(unittest.TestCase):
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

    def test_run_inference_success(self): 
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
        if img is None:
            return {"status": "error", "message": "Failed to decode image"}

        dimensions = self.assess_dimensions(img) 
        img = self.process(img)
        _, img_encoded = cv2.imencode('.jpg', img)
        image_bytes = img_encoded.tobytes()
        
        result = run_inference(image_bytes)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("class_id", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["backend"], "jax/flax")

    def test_run_inference_invalid_image(self):
        result = run_inference(b"not an image")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Could not decode image")

if __name__ == '__main__':
    unittest.main()
