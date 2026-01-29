import unittest
import numpy as np
import cv2
import sys
import os

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge.jax_train import run_inference

class TestJaxInference(unittest.TestCase):
    def test_run_inference_success(self):
        # Create a dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
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
