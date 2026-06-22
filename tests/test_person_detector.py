from __future__ import annotations

import unittest

import numpy as np

from services.vision_ai.src.person_detection import PersonDetector


class _Tensor:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values)

    def int(self) -> "_Tensor":
        return _Tensor(self._values.astype(int))

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = _Tensor([[1, 2, 30, 40], [5, 6, 15, 16], [0, 0, 9, 9]])
        self.conf = _Tensor([0.75, 0.95, 0.40])
        self.cls = _Tensor([0, 2, 0])


class _Result:
    def __init__(self, boxes: object) -> None:
        self.boxes = boxes


class _Model:
    def __init__(self, results: object = None) -> None:
        self.results = results if results is not None else [_Result(_Boxes())]
        self.last_kwargs = None

    def predict(self, **kwargs: object) -> object:
        self.last_kwargs = kwargs
        return self.results


class PersonDetectorTests(unittest.TestCase):
    def test_detect_filters_person_class_and_sorts_by_confidence(self) -> None:
        model = _Model()
        detector = PersonDetector(
            model_name="yolo11n",
            confidence_threshold=0.3,
            iou_threshold=0.6,
            input_size=320,
            device="cpu",
            model=model,
        )
        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        detections = detector.detect(frame)

        self.assertEqual([item.confidence for item in detections], [0.75, 0.4])
        self.assertEqual(detections[0].bbox, [1, 2, 30, 40])
        self.assertTrue(all(item.class_id == 0 for item in detections))
        self.assertEqual(model.last_kwargs["classes"], [0])
        self.assertEqual(model.last_kwargs["conf"], 0.3)
        self.assertEqual(model.last_kwargs["iou"], 0.6)
        self.assertEqual(model.last_kwargs["imgsz"], 320)
        self.assertEqual(model.last_kwargs["device"], "cpu")

    def test_detect_handles_empty_results(self) -> None:
        detector = PersonDetector(model=_Model([_Result(None)]))
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        self.assertEqual(detector.detect(frame), [])

    def test_rejects_invalid_frames_before_inference(self) -> None:
        detector = PersonDetector(model=_Model())

        invalid_frames = (
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 4), dtype=np.uint8),
            np.zeros((0, 10, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.float32),
        )
        for frame in invalid_frames:
            with self.subTest(shape=frame.shape, dtype=frame.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    detector.detect(frame)

        with self.assertRaises(TypeError):
            detector.detect("not-an-image")  # type: ignore[arg-type]

    def test_rejects_invalid_configuration_values(self) -> None:
        invalid_arguments = (
            {"confidence_threshold": -0.1},
            {"confidence_threshold": 1.1},
            {"iou_threshold": "0.5"},
            {"input_size": 0},
            {"input_size": 640.0},
            {"device": ""},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    PersonDetector(model=_Model(), **arguments)

    def test_rejects_unsupported_model_and_invalid_model_object(self) -> None:
        with self.assertRaises(ValueError):
            PersonDetector(model_name="unknown", model=_Model())
        with self.assertRaises(TypeError):
            PersonDetector(model=object())


if __name__ == "__main__":
    unittest.main()
