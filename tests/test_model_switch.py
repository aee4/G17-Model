"""Single deployed model: plain-language view by default; Exp-25 adapter kept for later comparisons (mocked weights)."""
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


class DeployedModel(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        file = BytesIO()
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 80), dtype=np.uint8)).save(file, format="PNG")
        file.name = "scan.png"
        self.files = [file]
        self.checker = Mock()
        self.checker.classify.return_value = Decision("US", "modality", .99)

    def run_app(self, final):
        final.review_rule = {"metric": "agreement", "direction": "below", "threshold": 0.9357}
        with patch("streamlit.file_uploader", return_value=self.files), \
                patch("modality_check.load_modality_check", return_value=self.checker), \
                patch("checkpoint.load_checkpoint", return_value=final) as final_load, \
                patch("checkpoint_exp25.load_exp25") as exp25_load:
            app = AppTest.from_file(APP_PATH, default_timeout=60).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            return app, final_load, exp25_load

    def test_no_model_selector_and_only_deployed_model_loads(self):
        final = Mock(side_effect=ensemble_forward)
        app, final_load, exp25_load = self.run_app(final)
        self.assertEqual(len(app.radio), 0)
        self.assertEqual(final_load.call_count, 1)
        exp25_load.assert_not_called()
        self.assertNotIn("compare_batch_data", app.session_state)

    def test_default_view_is_plain_language(self):
        app, _, _ = self.run_app(Mock(side_effect=ensemble_forward))
        self.assertEqual([t.label for t in app.tabs], ["Outline", "Original", "Probability", "Mask", "Model input"])
        labels = [m.label for m in app.metric]
        self.assertEqual(labels[:2], ["Lesion size", "Confidence"])
        self.assertIn(app.metric[1].value, {"High", "Moderate", "Low", "No lesion found"})
        self.assertEqual(list(app.dataframe[0].value.columns), ["File", "Lesion size (of image)", "Confidence", "Status"])
        app.toggle(key="show_technical").set_value(True).run()
        self.assertIn("Model agreement", list(app.dataframe[0].value.columns))

    def test_threshold_change_reuses_saved_results(self):
        final = Mock(side_effect=ensemble_forward)
        app, _, _ = self.run_app(final)
        app.slider[0].set_value(.7).run()
        self.assertEqual(final.call_count, 1)

    def test_exp25_adapter_contract(self):
        inner = Mock(return_value={"logits": torch.zeros(2, 1, 8, 8)})
        out = Exp25Model(inner)(pixel_values=torch.zeros(2, 1, 8, 8))
        self.assertEqual(tuple(out["member_probabilities"].shape), (2, 1, 8, 8))
        torch.testing.assert_close(out["probabilities"], torch.full((2, 1, 8, 8), .5))
        self.assertIsNone(Exp25Model.review_rule)


if __name__ == "__main__":
    unittest.main()
