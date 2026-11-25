"""Ensemble output contract, quality signals and fail-closed checkpoint loading (tiny local files, no downloads)."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import segmentation_models_pytorch as smp
import torch
from safetensors.torch import save_file

import checkpoint
from checkpoint import SegmentationEnsemble, load_checkpoint
from uncertainty import needs_review, quality_signals

RULE = {"metric": "agreement", "direction": "below", "threshold": 0.9}


def write_bundle(directory, mutate=None, members=2):
    names, hashes = [], {}
    for i in range(members):
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
        state = {k: v.contiguous() for k, v in model.state_dict().items()}
        if mutate and i == 0:
            state = mutate(state)
        name = f"member_{i}.safetensors"
        save_file(state, str(Path(directory) / name))
        names.append(name)
        hashes[name] = hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
    config = dict(architectures=["UnetPlusPlus"], encoder_name="resnet34", in_channels=1, classes=1,
                  image_size=256, members=names, sha256=hashes, review_rule=RULE)
    (Path(directory) / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return config


class CheckpointTests(unittest.TestCase):
    def test_ensemble_output_contract(self):
        members = [torch.nn.Conv2d(1, 1, 1) for _ in range(3)]
        out = SegmentationEnsemble(members, RULE, {})(pixel_values=torch.zeros(2, 1, 32, 32))
        self.assertEqual(tuple(out["probabilities"].shape), (2, 1, 32, 32))
        self.assertEqual(tuple(out["member_probabilities"].shape), (2, 3, 32, 32))
        torch.testing.assert_close(out["probabilities"][:, 0], out["member_probabilities"].mean(1))

    def test_loader_accepts_valid_bundle_and_rejects_bad_ones(self):
        cases = {
            "valid": None,
            "missing key": lambda s: {k: v for k, v in s.items() if not k.endswith("num_batches_tracked")},
            "extra key": lambda s: {**s, "unexpected.weight": torch.zeros(1)},
        }
        for label, mutate in cases.items():
            with self.subTest(case=label), TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {checkpoint.LOCAL_DIR_ENV: directory}):
                write_bundle(directory, mutate)
                if mutate is None:
                    model = load_checkpoint()
                    self.assertEqual(len(model.members), 2)
                    self.assertEqual(model.review_rule, RULE)
                else:
                    with self.assertRaises(RuntimeError):
                        load_checkpoint()

    def test_loader_rejects_tampered_weights_and_bad_config(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {checkpoint.LOCAL_DIR_ENV: directory}):
            config = write_bundle(directory)
            with open(Path(directory) / config["members"][1], "ab") as handle:
                handle.write(b"x")
            with self.assertRaises(RuntimeError):
                load_checkpoint()
            for key, value in (("encoder_name", "resnet50"), ("review_rule", {"metric": "agreement"}), ("members", [])):
                with self.subTest(key=key):
                    write_bundle(directory)
                    bad = json.loads((Path(directory) / "config.json").read_text())
                    bad[key] = value
                    (Path(directory) / "config.json").write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        load_checkpoint()


class QualitySignals(unittest.TestCase):
    def test_agreement_and_review(self):
        probs = np.zeros((3, 64, 64), np.float32)
        probs[:, 10:40, 10:40] = .9
        signals = quality_signals(probs)
        self.assertAlmostEqual(signals["agreement"], 1.0)
        self.assertFalse(needs_review(signals, RULE))
        probs[1] = 0
        self.assertTrue(needs_review(quality_signals(probs), RULE))
        self.assertTrue(needs_review(quality_signals(np.zeros((3, 8, 8))), RULE))  # empty mask always flagged


if __name__ == "__main__":
    unittest.main()
