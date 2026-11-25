"""Exercise the real app functions without starting Streamlit or downloading AMFU-Net."""
import ast
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
from PIL import Image
import torch

from modality_check import Decision, screen_uploads
from preprocessing import preprocess_ultrasound
from uncertainty import needs_review, quality_signals


class PipelineIntegration(unittest.TestCase):
    def test_accepted_pixels_and_tensor_reach_existing_pipeline_unchanged(self):
        source = ast.parse(Path("app.py").read_text(encoding="utf-8"))
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_preprocess_single", "run_batch_inference")]
        namespace = dict(ThreadPoolExecutor=ThreadPoolExecutor, Image=Image,
                         preprocess_ultrasound=preprocess_ultrasound, torch=torch,
                         cv2=cv2, quality_signals=quality_signals, needs_review=needs_review)
        exec(compile(ast.Module(body=functions, type_ignores=[]), "app.py", "exec"), namespace)
        pixels = np.random.default_rng(42).integers(0, 256, (80, 120, 3), dtype=np.uint8)
        file = BytesIO()
        Image.fromarray(pixels).save(file, format="PNG")
        file.name = "accepted.png"
        original_bytes = file.getvalue()
        expected = preprocess_ultrasound(Image.open(BytesIO(original_bytes)))
        checker = Mock()
        checker.classify.return_value = Decision("US", "modality", .99)
        accepted, skipped = screen_uploads([file], checker)
        self.assertFalse(skipped)
        self.assertEqual(accepted[0].getvalue(), original_bytes)

        stages = []

        def forward(*, pixel_values):
            self.assertEqual(stages[-1], "segmenting")
            torch.testing.assert_close(pixel_values, expected["tensor"], rtol=0, atol=0)
            return {"probabilities": torch.zeros((1, 1, 256, 256)),
                    "member_probabilities": torch.zeros((1, 3, 256, 256))}

        result = namespace["run_batch_inference"](accepted, forward, stages.append)[0]
        self.assertEqual(stages, ["enhancing", "segmenting", "preparing"])
        np.testing.assert_array_equal(result["original"], expected["original"])
        np.testing.assert_array_equal(result["model_input"], expected["model_input"])
        self.assertEqual(result["probability"].shape, (80, 120))
        self.assertIsNone(result["review"])  # plain callables carry no calibrated rule
        self.assertEqual(result["signals"]["area_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
