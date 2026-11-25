"""Load the G17 final segmentation ensemble with strict, verified weight loading.

Three U-Net++ (ResNet-34 encoder) members trained with the R5 recipe on train + validation
+ BrEaST. Each member is flip-averaged; the ensemble probability is the member mean.
Weights and config come from the final/ folder of a pinned Hugging Face revision, or from a local folder
named by the G17_MODEL_DIR environment variable (same files: config.json + member_*.safetensors).
"""
import hashlib
import json
import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


MODEL_REPOSITORY = "aee4/G17-BUS-Segmentation"
MODEL_SUBFOLDER = "final"          # r5_paper/ holds the reported (paper) ensemble
MODEL_BRANCH = "main"
MODEL_REVISION = "main"  # pinned to the upload commit hash once published
MODEL_LABEL = "G17 final: U-Net++ (ResNet-34) × 3 ensemble, flip TTA"
LOCAL_DIR_ENV = "G17_MODEL_DIR"

EXPECTED = dict(architectures=["UnetPlusPlus"], encoder_name="resnet34", in_channels=1, classes=1, image_size=256)
RULE_KEYS = {"metric", "direction", "threshold"}


class SegmentationEnsemble(torch.nn.Module):
    """forward(pixel_values) -> {"probabilities": (B,1,H,W), "member_probabilities": (B,M,H,W)}."""

    def __init__(self, members, review_rule, info):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
        self.review_rule = review_rule
        self.info = info

    @torch.inference_mode()
    def forward(self, pixel_values):
        probabilities = []
        for member in self.members:
            p = torch.sigmoid(member(pixel_values))
            flipped = torch.sigmoid(member(torch.flip(pixel_values, [-1])))
            probabilities.append(0.5 * (p + torch.flip(flipped, [-1]))[:, 0])
        stacked = torch.stack(probabilities, dim=1)
        return {"probabilities": stacked.mean(1, keepdim=True), "member_probabilities": stacked}


def _fetch(name):
    local = os.environ.get(LOCAL_DIR_ENV)
    if local:
        path = Path(local) / name
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found in {LOCAL_DIR_ENV}={local}")
        return str(path)
    return hf_hub_download(MODEL_REPOSITORY, f"{MODEL_SUBFOLDER}/{name}", revision=MODEL_REVISION)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint():
    """Return a CPU eval ensemble only after config, hashes and every weight key check out."""
    with open(_fetch("config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    for key, value in EXPECTED.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected model config {key}: {config.get(key)!r}")
    members = config.get("members")
    rule = config.get("review_rule")
    if not isinstance(members, list) or not members:
        raise ValueError("Model config lists no ensemble members")
    if not isinstance(rule, dict) or set(rule) != RULE_KEYS or rule["direction"] not in ("below", "above"):
        raise ValueError(f"Invalid review rule: {rule!r}")

    models = []
    for name in members:
        path = _fetch(name)
        expected_hash = config.get("sha256", {}).get(name)
        if expected_hash is None or _sha256(path) != expected_hash:
            raise RuntimeError(f"Checksum mismatch for {name}")
        model = smp.UnetPlusPlus(encoder_name=config["encoder_name"], encoder_weights=None,
                                 in_channels=config["in_channels"], classes=config["classes"])
        weights = load_file(path, device="cpu")
        expected_keys = set(model.state_dict())
        missing = sorted(expected_keys - weights.keys())
        unexpected = sorted(weights.keys() - expected_keys)
        if missing or unexpected:
            raise RuntimeError(f"{name} key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        model.load_state_dict(weights, strict=True)
        models.append(model.eval())
    info = {k: config.get(k) for k in ("internal_test", "review_rule_calibration", "training")}
    return SegmentationEnsemble(models, rule, info).eval()
