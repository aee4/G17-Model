"""Per-scan quality signals from the ensemble, and the calibrated "check this scan" rule.

All signals are computed on the 256 x 256 model grid from each member's flip-averaged probability map.
The rule (metric, direction, threshold) is fitted on the held-out internal test set and shipped with the
weights, so the app never invents its own cut-off.
"""
import numpy as np
from scipy.ndimage import label

SIGNALS = ("agreement", "ambiguity", "mean_confidence", "components", "area_fraction")


def _dice(a, b):
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


def quality_signals(member_probs, threshold=0.5):
    """member_probs: (members, H, W) probabilities. Returns a dict of scalar signals."""
    member_probs = np.asarray(member_probs, dtype=np.float32)
    mean = member_probs.mean(0)
    mask = mean >= threshold
    members = member_probs >= threshold
    pairs = [_dice(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))]
    area = int(mask.sum())
    return {
        "agreement": float(np.mean(pairs)) if pairs else 1.0,        # mean pairwise Dice between member masks
        "ambiguity": float(((mean > 0.3) & (mean < 0.7)).sum() / max(area, 1)),  # uncertain band / mask area
        "mean_confidence": float(mean[mask].mean()) if area else 0.0,
        "components": int(label(mask)[1]),
        "area_fraction": float(area / mask.size),
    }


def needs_review(signals, rule):
    """True when the scan should be checked by a person. An empty mask is always flagged."""
    if signals["area_fraction"] == 0:
        return True
    value = signals[rule["metric"]]
    return value < rule["threshold"] if rule["direction"] == "below" else value > rule["threshold"]
