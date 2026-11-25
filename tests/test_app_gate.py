"""Streamlit control-flow checks with mocked model responses, no weight downloads."""
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest
import torch

from modality_check import Decision

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class AppGate(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.files = []
        for i in range(2):
            file = BytesIO()
            Image.new("L", (64, 48), 100 + i).save(file, format="PNG")
            file.name = f"upload-{i}.png"
            self.files.append(file)

    def test_loading_failure_clears_old_results_and_blocks_model(self):
        with patch("streamlit.file_uploader", return_value=self.files), \
                patch("modality_check.load_modality_check", side_effect=OSError("offline")), \
                patch("checkpoint.load_checkpoint") as segmenter:
            app = AppTest.from_file(APP_PATH, default_timeout=20).run()
            app.session_state["raw_batch_data"] = ["stale"]
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn("Segmentation blocked", app.error[0].value)
            self.assertNotIn("raw_batch_data", app.session_state)
            segmenter.assert_not_called()

    def test_all_rejected_batch_does_not_load_segmenter(self):
        checker = Mock()
        checker.classify.return_value = Decision("CT", "modality", .99)
        with patch("streamlit.file_uploader", return_value=self.files), \
                patch("modality_check.load_modality_check", return_value=checker), \
                patch("checkpoint.load_checkpoint") as segmenter:
            app = AppTest.from_file(APP_PATH, default_timeout=20).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["skipped_files"]), 2)
            self.assertNotIn("raw_batch_data", app.session_state)
            self.assertTrue(any("No images passed" in item.value for item in app.info))
            segmenter.assert_not_called()

    def test_classification_failure_blocks_previously_accepted_file(self):
        checker = Mock()
        checker.classify.side_effect = [Decision("US", "modality", .99), RuntimeError("bad graph")]
        with patch("streamlit.file_uploader", return_value=self.files), \
                patch("modality_check.load_modality_check", return_value=checker), \
                patch("checkpoint.load_checkpoint") as segmenter:
            app = AppTest.from_file(APP_PATH, default_timeout=20).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn("Segmentation blocked", app.error[0].value)
            segmenter.assert_not_called()

    def test_mixed_batch_and_threshold_rerun(self):
        checker = Mock()
        checker.classify.side_effect = [Decision("US", "modality", .99), Decision("CT", "modality", .99)]
        model = Mock(return_value={"probabilities": torch.zeros((1, 1, 256, 256)),
                                    "member_probabilities": torch.zeros((1, 3, 256, 256))})
        with patch("streamlit.file_uploader", return_value=self.files), \
                patch("modality_check.load_modality_check", return_value=checker), \
                patch("checkpoint.load_checkpoint", return_value=model):
            app = AppTest.from_file(APP_PATH, default_timeout=20).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state["raw_batch_data"]), 1)
            self.assertEqual(len(app.session_state["skipped_files"]), 1)
            app.slider[0].set_value(.7).run()
            self.assertFalse(app.exception)
            self.assertEqual(checker.classify.call_count, 2)
            self.assertEqual(model.call_count, 1)
            app.button[0].click().run()  # Exhausted mock classifier now fails.
            self.assertFalse(app.exception)
            self.assertNotIn("raw_batch_data", app.session_state)


if __name__ == "__main__":
    unittest.main()
