import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from edge.vision_module import WebcamStream, ObjectDetector

class TestVisionModule(unittest.TestCase):

    @patch('cv2.VideoCapture')
    def test_webcam_stream_initialization(self, mock_vc):
        mock_vc.return_value.isOpened.return_value = True
        stream = WebcamStream(src=0)
        self.assertTrue(mock_vc.called)
        stream.release()

    @patch('cv2.VideoCapture')
    def test_webcam_stream_get_frame(self, mock_vc):
        mock_vc.return_value.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_vc.return_value.read.return_value = (True, dummy_frame)
        
        stream = WebcamStream(src=0)
        frame = stream.get_frame()
        
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        stream.release()

    @patch('edge.vision_module.fasterrcnn_resnet50_fpn')
    @patch('edge.vision_module.FasterRCNN_ResNet50_FPN_Weights')
    def test_object_detector_logic(self, mock_weights_class, mock_model_fn):
        # Mocking the model and its inference
        mock_model = MagicMock()
        mock_model_fn.return_value = mock_model
        
        # Mock prediction output
        mock_prediction = {
            'labels': torch.tensor([1]),
            'scores': torch.tensor([0.9]),
            'boxes': torch.tensor([[10, 10, 100, 100]])
        }
        mock_model.return_value = [mock_prediction]
        
        # Mock the weights bit to avoid errors
        mock_weights = MagicMock()
        mock_weights.meta = {"categories": ["__background__", "person"]}
        mock_weights_class.DEFAULT = mock_weights

        detector = ObjectDetector(threshold=0.5)
        # Note: categories is set in __init__ from mock_weights.meta
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        prediction = detector.detect(dummy_frame)
        self.assertEqual(len(prediction['labels']), 1)
        
        labels = detector.get_labels(prediction)
        self.assertEqual(labels[0][0], "person")
        self.assertAlmostEqual(labels[0][1], 0.9)

if __name__ == '__main__':
    unittest.main()
