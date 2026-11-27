"""Model selector: Exp-25 path keeps SRAD/CLAHE views, Compare runs both models once each (mocked weights)."""
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest
import torch

from checkpoint_exp25 import Exp25Model
from modality_check import Decision

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def ensemble_forward(*, pixel_values):
    p = torch.zeros(len(pixel_values), 1, 256, 256); p[:, :, 64:192, 64:192] = .9
    return {"probabilities": p, "member_probabilities": p.repeat(1, 3, 1, 1)}


def exp25_forward(*, pixel_values):
    p = torch.zeros(len(pixel_values), 1, 256, 256); p[:, :, 96:224, 64:192] = .9
    return {"probabilities": p, "member_probabilities": p}


class ModelSwitch(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        file = BytesIO()
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 80), dtype=np.uint8)).save(file, format="PNG")
        file.name = "scan.png"
        self.files = [file]
        self.checker = Mock()
        self.checker.classify.return_value = Decision("US", "modality", .99)

    def run_choice(self, choice, final, exp25):
        final.review_rule = {"metric": "agreement", "direction": "below", "threshold": 0.9357}
        exp25.review_rule = None
        with patch("streamlit.file_uploader", return_value=self.files), \
                patch("modality_check.load_modality_check", return_value=self.checker), \
                patch("checkpoint.load_checkpoint", return_value=final) as final_load, \
                patch("checkpoint_exp25.load_exp25", return_value=exp25) as exp25_load:
            app = AppTest.from_file(APP_PATH, default_timeout=60).run()
            app.radio(key="model_choice").set_value(choice).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            return app, final_load, exp25_load

    def test_exp25_only_loads_legacy_model_and_shows_srad_tabs(self):
        final, exp25 = Mock(side_effect=ensemble_forward), Mock(side_effect=exp25_forward)
        app, final_load, exp25_load = self.run_choice("exp25", final, exp25)
        final_load.assert_not_called()
        self.assertEqual(exp25_load.call_count, 1)
        self.assertEqual(app.session_state["analysis_models"], ["exp25"])
        self.assertIn("enhanced", app.session_state["raw_batch_data"][0])
        self.assertEqual([t.label for t in app.tabs],
                         ["Outline", "Original", "Probability", "Mask", "Model input", "SRAD + CLAHE", "SRAD only"])
        self.assertIsNone(app.session_state["raw_batch_data"][0]["review"])

    def test_compare_runs_each_model_once_and_reports_overlap(self):
        final, exp25 = Mock(side_effect=ensemble_forward), Mock(side_effect=exp25_forward)
        app, final_load, exp25_load = self.run_choice("compare", final, exp25)
        self.assertEqual((final.call_count, exp25.call_count), (1, 1))
        self.assertEqual(app.session_state["analysis_models"], ["final", "exp25"])
        self.assertIn("compare_batch_data", app.session_state)
        overlap = [m for m in app.metric if "Overlap" in m.label]
        self.assertEqual(len(overlap), 1)
        self.assertTrue(overlap[0].value.endswith("%"))
        app.slider[0].set_value(.7).run()   # threshold changes reuse both saved results
        self.assertEqual((final.call_count, exp25.call_count), (1, 1))

    def test_default_view_is_plain_language(self):
        final, exp25 = Mock(side_effect=ensemble_forward), Mock(side_effect=exp25_forward)
        app, _, _ = self.run_choice("final", final, exp25)
        self.assertEqual([t.label for t in app.tabs][:2], ["Outline", "Original"])
        labels = [m.label for m in app.metric]
        self.assertEqual(labels[:2], ["Lesion size", "Confidence"])
        self.assertIn(app.metric[1].value, {"High", "Moderate", "Low", "No lesion found"})
        self.assertEqual(list(app.dataframe[0].value.columns), ["File", "Lesion size (of image)", "Confidence", "Status"])
        app.toggle(key="show_technical").set_value(True).run()
        self.assertIn("Model agreement", list(app.dataframe[0].value.columns))

    def test_switching_model_clears_previous_results(self):
        final, exp25 = Mock(side_effect=ensemble_forward), Mock(side_effect=exp25_forward)
        app, _, _ = self.run_choice("final", final, exp25)
        self.assertIn("raw_batch_data", app.session_state)
        app.radio(key="model_choice").set_value("exp25").run()
        self.assertNotIn("raw_batch_data", app.session_state)

    def test_exp25_adapter_contract(self):
        inner = Mock(return_value={"logits": torch.zeros(2, 1, 8, 8)})
        out = Exp25Model(inner)(pixel_values=torch.zeros(2, 1, 8, 8))
        self.assertEqual(tuple(out["member_probabilities"].shape), (2, 1, 8, 8))
        torch.testing.assert_close(out["probabilities"], torch.full((2, 1, 8, 8), .5))
        self.assertIsNone(Exp25Model.review_rule)


if __name__ == "__main__":
    unittest.main()
