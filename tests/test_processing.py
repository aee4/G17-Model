"""Verify real execution boundaries, cached-model behavior and terminal UI states."""
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest
import torch

from modality_check import Decision
from model_resource import ModelResource
from workspace_ui import ProcessingState, processing_html

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class ProcessingLifecycle(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.files = []
        for value in (80, 120):
            file = BytesIO()
            Image.new("L", (80, 64), value).save(file, format="PNG")
            file.name = "duplicate.png"
            self.files.append(file)
        self.events = []

    def render(self, state):
        self.events.append((state.stage, state.status))
        return processing_html(state)

    def test_stages_follow_operations_and_cached_model_skips_loading(self):
        def classify(image):
            self.assertEqual(self.events[-1], ("checking", "running"))
            return Decision("US", "modality", .99)

        def forward(*, pixel_values):
            self.assertEqual(self.events[-1], ("segmenting", "running"))
            return {"probabilities": torch.zeros(len(pixel_values), 1, 256, 256),
                    "member_probabilities": torch.zeros(len(pixel_values), 3, 256, 256)}

        checker = Mock()
        checker.classify.side_effect = classify
        model = Mock(side_effect=forward)

        def factory():
            self.assertEqual(self.events[-1], ("loading", "running"))
            return model

        with patch("streamlit.file_uploader", return_value=self.files), \
                patch("workspace_ui.processing_html", side_effect=self.render), \
                patch("modality_check.load_modality_check", return_value=checker) as checker_load, \
                patch("checkpoint.load_checkpoint", side_effect=factory) as model_load:
            app = AppTest.from_file(APP_PATH, default_timeout=20).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            state = app.session_state["analysis_state"]
            self.assertEqual(state.visited, ["checking", "loading", "enhancing", "segmenting", "preparing", "complete"])
            self.assertNotIn("is-running", processing_html(state))
            self.assertEqual(state.status, "complete")
            app.selectbox[0].set_value(1).run()
            self.assertEqual(app.selectbox[0].value, 1)
            # Duplicate filenames select distinct scans by index.
            self.assertEqual(app.session_state["raw_batch_data"][1]["original"][0, 0], 120)
            app.slider[0].set_value(.7).run()
            self.assertEqual(model.call_count, 1)
            self.assertEqual(checker_load.call_count, 1)
            self.assertEqual(model_load.call_count, 1)
            self.assertEqual(checker.classify.call_count, 2)
            self.events.clear()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertNotIn(("loading", "running"), self.events)
            self.assertNotIn("loading", app.session_state["analysis_state"].visited)
            self.assertIn("Already cached", processing_html(app.session_state["analysis_state"]))
            self.assertEqual(model_load.call_count, 1)

    def test_all_rejected_never_enters_segmentation(self):
        checker = Mock()
        checker.classify.return_value = Decision("CT", "modality", .99)
        with patch("streamlit.file_uploader", return_value=self.files), \
                patch("modality_check.load_modality_check", return_value=checker), \
                patch("checkpoint.load_checkpoint") as factory:
            app = AppTest.from_file(APP_PATH, default_timeout=20).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            state = app.session_state["analysis_state"]
            self.assertEqual(state.status, "rejected")
            self.assertEqual(state.visited, ["checking"])
            self.assertNotIn("is-running", processing_html(state))
            self.assertNotIn("raw_batch_data", app.session_state)
            factory.assert_not_called()

    def test_model_loading_and_inference_failures_end_animation_clear_results(self):
        for fail_loading in (True, False):
            with self.subTest(fail_loading=fail_loading):
                st.cache_resource.clear()
                checker = Mock()
                checker.classify.return_value = Decision("US", "modality", .99)
                factory = Mock(side_effect=OSError("weights unavailable")) if fail_loading else Mock(
                    return_value=Mock(side_effect=RuntimeError("forward failed")))
                with patch("streamlit.file_uploader", return_value=self.files), \
                        patch("modality_check.load_modality_check", return_value=checker), \
                        patch("checkpoint.load_checkpoint", factory):
                    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
                    app.session_state["raw_batch_data"] = ["stale"]
                    app.button[0].click().run()
                    self.assertFalse(app.exception)
                    state = app.session_state["analysis_state"]
                    self.assertEqual(state.status, "failed")
                    self.assertEqual(state.stage, "loading" if fail_loading else "segmenting")
                    self.assertNotIn("is-running", processing_html(state))
                    self.assertNotIn("raw_batch_data", app.session_state)
                    self.assertIn("Segmentation failed", app.error[0].value)

    def test_resource_failure_can_retry_without_caching_partial_model(self):
        resource = ModelResource()
        observer = Mock()
        factory = Mock(side_effect=[OSError("offline"), object()])
        with self.assertRaises(OSError):
            resource.get(factory, observer)
        model = resource.get(factory, observer)
        self.assertIs(resource.get(factory, observer), model)
        self.assertEqual(observer.call_count, 2)
        self.assertEqual(factory.call_count, 2)

    def test_terminal_state_cannot_reactivate_and_skipped_steps_are_explicit(self):
        state = ProcessingState()
        state.finish("failed")
        with self.assertRaises(ValueError):
            state.advance("segmenting")
        self.assertIn("Not run", processing_html(state))
