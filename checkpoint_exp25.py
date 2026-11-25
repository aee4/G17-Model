"""Legacy Exp-25 AMFU-Net (U-Net + DAG, SRAD + CLAHE input), kept for side-by-side comparison in the app.

Loaded strictly from its pinned Hugging Face revision, exactly as the app shipped it before the final ensemble."""
import json

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file



MODEL_REPOSITORY = "Ahorts/G17-AMFU-Net-Paper"
MODEL_BRANCH = "exp25-unet-dag-gpu-reflect-enhanced-cosine-only"
MODEL_REVISION = "f2f5d8c9d95611360a5a991e74a9e8f2932b97a8"
MODEL_LABEL = "Exp-25: U-Net + DAG, SRAD + CLAHE"


def load_checkpoint():
    """Return a CPU eval model only after every safetensors key loads strictly."""
    from modeling_amfu import AMFUConfig, AMFUNet  # lazy: transformers is only needed for this legacy model
    config_path = hf_hub_download(
        MODEL_REPOSITORY, "config.json", revision=MODEL_REVISION
    )
    with open(config_path, encoding="utf-8") as handle:
        values = json.load(handle)
    expected = dict(architectures=["AMFUNet"], base_channels=32,
                    image_size=256, use_dag=True, use_mfef=False)
    for key, value in expected.items():
        if key not in values or type(values[key]) is not type(value) or values[key] != value:
            raise ValueError(f"Unexpected Exp-25 config {key}: {values.get(key)!r}")

    weights_path = hf_hub_download(
        MODEL_REPOSITORY, "model.safetensors", revision=MODEL_REVISION
    )
    model = AMFUNet(AMFUConfig(**values))
    # Check all keys explicitly: PyTorch permits legacy checkpoints to omit
    # BatchNorm num_batches_tracked buffers even with strict=True.
    weights = load_file(weights_path, device="cpu")
    expected_keys = set(model.state_dict())
    missing = sorted(expected_keys - weights.keys())
    unexpected = sorted(weights.keys() - expected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


class Exp25Model:
    """Adapts AMFU-Net to the app's model contract: probabilities + one 'member' map, no review rule."""
    review_rule = None

    def __init__(self, model):
        self.model = model

    def __call__(self, *, pixel_values):
        import torch
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(pixel_values=pixel_values)["logits"])
        return {"probabilities": probabilities, "member_probabilities": probabilities[:, :1]}


def load_exp25():
    return Exp25Model(load_checkpoint())
