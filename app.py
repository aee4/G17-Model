from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from workspace_ui import ProcessingState, processing_html, header_html
from model_resource import ModelResource
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

import checkpoint
import checkpoint_exp25
from checkpoint import MODEL_BRANCH, MODEL_LABEL, MODEL_REPOSITORY, MODEL_REVISION
from preprocessing import preprocess_exp25, preprocess_ultrasound
from uncertainty import needs_review, quality_signals
from modality_check import load_modality_check, screen_uploads


# Models offered in the interface. Loaders are looked up at call time so tests can patch the modules.
MODELS = {
    "final": dict(label="G17 final (U-Net++ ensemble)", dice="89.06%",
                  loader=lambda: checkpoint.load_checkpoint(), preprocess=preprocess_ultrasound),
    "exp25": dict(label="Exp-25 (AMFU-Net)", dice="82.76%",
                  loader=lambda: checkpoint_exp25.load_exp25(), preprocess=preprocess_exp25),
}
MODEL_CHOICES = {"final": "Current model", "exp25": "Previous model", "compare": "Compare both"}
PLAIN_NAMES = {"final": "the current model", "exp25": "the previous model"}

# Plain-language labels shown by default. Raw values stay available under "Advanced details".
CONFIDENCE_BANDS = ((.85, "High"), (.70, "Moderate"))   # on mean probability inside the mask; not calibrated
CONFIDENCE_HELP = ("Average model certainty inside the outline: High at 85% or more, Moderate at 70–85%, Low below "
                   "70%. These bands are a reading aid, not a calibrated accuracy estimate.")


def confidence_label(mean_conf, detected=True):
    if not detected:
        return "No lesion found"
    return next((label for cut, label in CONFIDENCE_BANDS if mean_conf >= cut), "Low")


def size_text(area_pct):
    if area_pct <= 0:
        return "No lesion found"
    return "Under 1%" if area_pct < 1 else f"About {round(area_pct)}%"


def status_text(item):
    """(status, reason) for the default view; status is None when the model has no Check rule."""
    if item.get("review") is None:
        return None, ""
    if not item["review"]:
        return "Reliable", "The three models agree closely on this outline."
    if item["signals"]["area_fraction"] == 0:
        return "Check", "No lesion was found. Look at the scan yourself."
    return "Check", "The three models disagree on where the lesion edge is. Look at the outline closely."


def clear_analysis():
    for key in ("raw_batch_data", "compare_batch_data", "analysis_models", "skipped_files",
                "input_check_error", "analysis_state"):
        st.session_state.pop(key, None)


@st.cache_resource(show_spinner=False)
def load_input_check():
    return load_modality_check()


st.set_page_config(page_title="G17 lesion segmentation", page_icon="assets/scan-line.svg", layout="wide")


@st.cache_resource(show_spinner=False)
def model_resource(key="final"):
    return ModelResource()


def load_model(key="final", _on_load=None):
    """Download each model once and reuse it across predictions."""
    return model_resource(key).get(MODELS[key]["loader"], _on_load)


AMBER, BLUE, IVORY = (233, 164, 72), (138, 184, 230), (237, 233, 226)


def _outline(image, mask, colour):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, colour, max(2, round(min(mask.shape) / 220)), lineType=cv2.LINE_AA)


def difference_overlay(original_gray, mask_a, mask_b):
    """Ivory = both models, amber = final model only, blue = Exp-25 only."""
    rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    a, b = mask_a > 0, mask_b > 0
    colour = np.zeros_like(rgb)
    colour[a & b], colour[a & ~b], colour[~a & b] = IVORY, AMBER, BLUE
    blended = np.where((a | b)[..., None], 0.4 * rgb + 0.6 * colour, rgb).clip(0, 255).astype(np.uint8)
    return blended


def mask_dice(mask_a, mask_b):
    a, b = mask_a > 0, mask_b > 0
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


def create_overlay(original_gray: np.ndarray, binary_mask: np.ndarray, colour=AMBER) -> np.ndarray:
    """Tint the predicted lesion and draw its outline over the ultrasound (amber for the deployed model)."""
    overlay = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB)
    region = binary_mask > 0
    tinted = (0.72 * overlay + 0.28 * np.array(colour, np.float32)).astype(np.uint8)
    overlay[region] = tinted[region]
    _outline(overlay, binary_mask, colour)
    return overlay


def _preprocess_single(file, preprocess=None):
    """Preprocess a single uploaded ultrasound file."""
    img = Image.open(file)
    processed = (preprocess or preprocess_ultrasound)(img)
    processed["filename"] = file.name
    return processed


def run_batch_inference(uploaded_files, model, stage_callback=None, preprocess=None):
    """Preprocess all images concurrently, execute a single batched forward pass, and store probability maps."""
    if stage_callback:
        stage_callback("enhancing")
    with ThreadPoolExecutor() as executor:
        processed_items = list(executor.map(lambda file: _preprocess_single(file, preprocess), uploaded_files))

    tensors = [item["tensor"].squeeze(0) for item in processed_items]
    batch_tensor = torch.stack(tensors, dim=0).to("cpu")

    if stage_callback:
        stage_callback("segmenting")
    with torch.inference_mode():
        output = model(pixel_values=batch_tensor)
        probabilities = output["probabilities"]
        member_probabilities = output["member_probabilities"]
    rule = getattr(model, "review_rule", None)
    rule = rule if isinstance(rule, dict) else None   # only a calibrated rule (dict) can flag scans

    if stage_callback:
        stage_callback("preparing")
    results = []
    for i, item in enumerate(processed_items):
        prob_map_256 = probabilities[i, 0].detach().cpu().numpy()
        h, w = item["original_size"]
        prob_map = cv2.resize(prob_map_256, (w, h), interpolation=cv2.INTER_LINEAR)
        # Quality signals use the model grid and the calibrated 0.5 cut-off, independent of the slider.
        signals = quality_signals(member_probabilities[i].detach().cpu().numpy())
        results.append(
            {
                "filename": item["filename"],
                "original": item["original"],
                "model_input": item["model_input"],
                **{key: item[key] for key in ("enhanced", "despeckled") if key in item},
                "probability": prob_map,
                "original_size": (h, w),
                "signals": signals,
                "review": bool(needs_review(signals, rule)) if rule else None,
            }
        )

    return results


def compute_derived_results(batch_results, threshold: float):
    """Derive binary masks, overlays, and summary metrics from cached probability maps."""
    derived = []
    summary_rows = []

    for item in batch_results:
        prob = item["probability"]
        h, w = item["original_size"]
        binary_mask = (prob >= float(threshold)).astype(np.uint8) * 255
        overlay = create_overlay(item["original"], binary_mask)

        lesion_pixels = int(np.count_nonzero(binary_mask))
        total_pixels = h * w
        area_pct = (lesion_pixels / total_pixels) * 100.0
        mean_conf = float(prob[binary_mask > 0].mean()) if lesion_pixels > 0 else 0.0

        derived.append(
            {
                **item,
                "mask": binary_mask,
                "overlay": overlay,
                "lesion_pixels": lesion_pixels,
                "area_pct": area_pct,
                "mean_conf": mean_conf,
                "lesion_detected": lesion_pixels > 0,
            }
        )

        status, _ = status_text(item)
        summary_rows.append(
            {
                "File": item["filename"],
                "Lesion size (of image)": size_text(area_pct),
                "Confidence": confidence_label(mean_conf, lesion_pixels > 0),
                "Status": status or "Not checked",
                "Image size": f"{w} × {h}",
                "Lesion area": f"{area_pct:.2f}%",
                "Lesion pixels": f"{lesion_pixels:,}",
                "Mean probability": f"{mean_conf * 100:.1f}%" if lesion_pixels > 0 else "No lesion",
            }
            | ({"Model agreement": f"{item['signals']['agreement'] * 100:.1f}%"} if item.get("review") is not None else {})
        )

    return derived, pd.DataFrame(summary_rows)


st.html(f"<style>{(Path(__file__).parent / 'assets/workspace.css').read_text(encoding='utf-8')}</style>")
st.html(header_html())
st.html('<p class="g17-notice">The models assume each scan contains a lesion. Results are not a diagnosis. '
        'Upload de-identified images only.</p>')

with st.container(key="upload_workspace"):
    st.html('<h2 class="g17-h2">Scans</h2>')
    model_choice = st.radio(
        "Model", options=list(MODEL_CHOICES), format_func=MODEL_CHOICES.get, horizontal=True,
        key="model_choice", on_change=clear_analysis,
        help="Current model: G17 U-Net++ ensemble (89.06% Dice on 266 held-out test scans). Previous model: Exp-25 "
             "AMFU-Net (82.76%). Compare both runs the two on the same scans.",
    )
    uploaded_files = st.file_uploader(
        "Breast ultrasound scans", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True, on_change=clear_analysis,
        help="PNG, JPEG, BMP or TIFF. Maximum 20 MB per file. Use de-identified images only.",
    )
    if uploaded_files:
        with st.container(horizontal=True, gap="small"):
            for index, file in enumerate(uploaded_files[:4]):
                with st.container(width=130):
                    try:
                        with Image.open(BytesIO(file.getvalue())) as preview:
                            preview.thumbnail((130, 95))
                            st.image(preview, width="content", alt=f"Uploaded scan {index + 1}")
                    except Exception:
                        st.caption("Preview unavailable")
                    st.text(file.name)
        if len(uploaded_files) > 4:
            st.caption(f"Showing 4 of {len(uploaded_files)} uploads. All of them will be analyzed.")
        run_button = st.button(
            f"Analyze {len(uploaded_files)} scan{'s' if len(uploaded_files) > 1 else ''}",
            type="primary",
        )
    else:
        run_button = False
    st.caption("Each upload is first checked to look like an ultrasound image. The check can't confirm breast "
               "anatomy or that a lesion is present, so an unrelated image can still pass.")

processing_slot = st.empty()
prepared = None
if run_button:
    clear_analysis()
    state = ProcessingState()
    st.session_state["analysis_state"] = state

    def present_stage(stage):
        state.advance(stage)
        processing_slot.html(processing_html(state))

    # Send the card before any model download, screening or segmentation work.
    processing_slot.html(processing_html(state))
    try:
        checker = load_input_check()
        accepted_files, skipped = screen_uploads(uploaded_files, checker)
        st.session_state["skipped_files"] = skipped
    except Exception as err:
        state.finish("failed")
        st.session_state["input_check_error"] = (
            f"Ultrasound input check unavailable or failed. Segmentation blocked. {err}"
        )
    else:
        state.accepted, state.skipped = len(accepted_files), len(skipped)
        processing_slot.html(processing_html(state))
        if accepted_files:
            try:
                keys = ["final", "exp25"] if model_choice == "compare" else [model_choice]
                models = {key: load_model(key, _on_load=present_stage) for key in keys}
                batch_data = run_batch_inference(accepted_files, models[keys[0]], stage_callback=present_stage,
                                                 preprocess=MODELS[keys[0]]["preprocess"])
                if len(keys) > 1:
                    st.session_state["compare_batch_data"] = run_batch_inference(
                        accepted_files, models[keys[1]], preprocess=MODELS[keys[1]]["preprocess"])
                prepared = compute_derived_results(batch_data, st.session_state.get("threshold", .5))
                st.session_state["raw_batch_data"] = batch_data
                st.session_state["analysis_models"] = keys
                present_stage("complete")
            except Exception as err:
                st.session_state.pop("raw_batch_data", None)
                st.session_state.pop("compare_batch_data", None)
                prepared = None
                state.finish("failed")
                st.session_state["input_check_error"] = f"Segmentation failed: {err}"
        else:
            state.finish("rejected")
    processing_slot.html(processing_html(state))
elif "analysis_state" in st.session_state:
    processing_slot.html(processing_html(st.session_state["analysis_state"]))

if st.session_state.get("input_check_error"):
    st.error(st.session_state["input_check_error"])
state = st.session_state.get("analysis_state")
if state is not None and state.status == "rejected":
    st.info("No images passed the ultrasound check, so nothing was segmented. Try a different scan.")

with st.container(key="inspection_workspace"):
    st.html('<h2 class="g17-h2">Results</h2>')
    if "raw_batch_data" not in st.session_state:
        st.html('<p class="g17-empty">Analyze scans to see the outlined lesion and whether each outline needs '
                'a closer look.</p>')
    else:
        model_keys = st.session_state.get("analysis_models", ["final"])
        primary = MODELS[model_keys[0]]
        n_scans = len(st.session_state["raw_batch_data"])
        flagged = sum(bool(r.get("review")) for r in st.session_state["raw_batch_data"])
        run_line = f"<b>{n_scans} scan{'s' if n_scans != 1 else ''}</b> analyzed with {PLAIN_NAMES[model_keys[0]]}"
        if "compare_batch_data" in st.session_state:
            run_line += f" and {PLAIN_NAMES[model_keys[1]]}"
        if flagged:
            run_line += f". <em>{flagged} need{'s' if flagged == 1 else ''} checking</em>"
        st.html(f'<p class="g17-run">{run_line}.</p>')
        viewer, rail = st.columns([5, 2], gap="large")
        with rail:
            with st.container(key="result_rail"):
                # Indices are identities: even identical filenames remain independently selectable.
                names = [f"{i + 1}. {r['filename']}" for i, r in enumerate(st.session_state["raw_batch_data"])]
                selected_idx = st.selectbox("Scan", options=list(range(n_scans)), format_func=names.__getitem__)
                stats_slot = st.container(key="result_rail_stats")
                advanced = st.expander("Advanced details")
                with advanced:
                    threshold = st.slider(
                        "Mask threshold", min_value=.10, max_value=.90, value=.50, step=.05, key="threshold",
                        help="Lower gives a larger outline, higher a tighter one. Default 0.50. Changing it reuses "
                             "the saved results, so nothing is re-run.",
                    )
        derived_results, summary_df = compute_derived_results(st.session_state["raw_batch_data"], threshold)
        compare_results = None
        if "compare_batch_data" in st.session_state:
            compare_results, _ = compute_derived_results(st.session_state["compare_batch_data"], threshold)
        item = derived_results[selected_idx]
        h, w = item["original_size"]

        status, reason = status_text(item)
        with stats_slot:
            if status == "Check":
                st.html(f'<div class="g17-flag"><b>Please check this outline.</b> {reason}</div>')
            elif status == "Reliable":
                st.html(f'<div class="g17-ok"><b>Outline looks reliable.</b> {reason}</div>')
            else:
                st.html('<div class="g17-ok">No reliability check for the previous model. It is a single network, '
                        'so there is nothing to compare its outline against.</div>')
            if compare_results:
                other = compare_results[selected_idx]
                st.metric("Lesion size, current model", size_text(item["area_pct"]), help="Share of the image inside the outline.")
                st.metric("Lesion size, previous model", size_text(other["area_pct"]), help="Share of the image inside the outline.")
                st.metric("Overlap between the two outlines", f"{mask_dice(item['mask'], other['mask']) * 100:.0f}%",
                          help="100% means both models outlined exactly the same area.")
            else:
                st.metric("Lesion size", size_text(item["area_pct"]), help="Share of the image inside the outline.")
                st.metric("Confidence", confidence_label(item["mean_conf"], item["lesion_detected"]),
                          help=CONFIDENCE_HELP)
        with advanced:
            st.metric("Lesion area", f"{item['area_pct']:.2f}%", help=f"{item['lesion_pixels']:,} pixels of {w} × {h}.")
            st.metric("Mean probability inside the mask",
                      f"{item['mean_conf'] * 100:.1f}%" if item["lesion_detected"] else "No lesion found")
            if item.get("review") is not None:
                st.metric("Model agreement", f"{item['signals']['agreement'] * 100:.1f}%",
                          help="Mean Dice overlap between the three ensemble members' masks.")
                st.metric("Separate regions", f"{item['signals']['components']}")
                st.caption("Check rule: flag when model agreement is below 93.57% or the mask is empty (both at the "
                           "0.50 cut-off). On 266 held-out test scans it flagged 12% and caught 7 of 8 failed outlines.")
            if compare_results:
                st.metric("Lesion area, previous model", f"{other['area_pct']:.2f}%")

        with viewer:
            if compare_results:
                other = compare_results[selected_idx]
                side, difference, original = st.tabs(["Side by side", "Differences", "Original"])
                with side:
                    left, right = st.columns(2, gap="small")
                    left.image(item["overlay"], caption="Current model",
                               width="stretch", alt="G17 final model outline")
                    right.image(create_overlay(other["original"], other["mask"], BLUE),
                                caption="Previous model", width="stretch",
                                alt="Exp-25 model outline")
                with difference:
                    st.image(difference_overlay(item["original"], item["mask"], other["mask"]), width="stretch",
                             alt="Where the two models agree and differ")
                    st.html('<div class="g17-legend"><span><i style="background:#EDE9E2"></i>Both models</span>'
                            '<span><i style="background:#E9A448"></i>Current model only</span>'
                            '<span><i style="background:#8AB8E6"></i>Previous model only</span></div>')
                with original:
                    st.image(item["original"], width="stretch", alt="Original ultrasound scan")
                st.caption("The previous model sees the scan after noise smoothing and contrast enhancement; the "
                           "current model sees the original image.")
            else:
                extra = [k for k in ("enhanced", "despeckled") if k in item]
                outline_tab, original_tab = st.tabs(["Outline", "Original"])
                with outline_tab:
                    st.image(item["overlay"], width="stretch", alt="Scan with the predicted lesion outlined")
                with original_tab:
                    st.image(item["original"], width="stretch", alt="Original ultrasound scan")
                with st.expander("Advanced details: probability map, mask and model input"):
                    names = ["Probability", "Mask", "Model input"] + \
                            [{"enhanced": "SRAD + CLAHE", "despeckled": "SRAD only"}[k] for k in extra]
                    tabs = st.tabs(names)
                    with tabs[0]:
                        st.image(item["probability"], caption="Brighter means the model is more sure the pixel is lesion.",
                                 width="stretch", clamp=True, alt="Lesion probability map")
                    with tabs[1]:
                        st.image(item["mask"], caption=f"Pixels at or above {threshold:.2f}", width="stretch",
                                 alt="Binary lesion mask")
                    with tabs[2]:
                        caption = ("256 × 256 input after SRAD + CLAHE" if extra else
                                   "256 × 256 grayscale input. This model uses no despeckling or contrast enhancement.")
                        st.image(item["model_input"], caption=caption, width="stretch", alt="Model input image")
                    for tab, key in zip(tabs[3:], extra):
                        with tab:
                            st.image(item[key], width="stretch", alt=f"{key} image")

        basic = ["File", "Lesion size (of image)", "Confidence", "Status"]
        if compare_results:
            summary_df["Previous model size (of image)"] = [size_text(r["area_pct"]) for r in compare_results]
            summary_df["Overlap with previous model"] = [f"{mask_dice(a['mask'], b['mask']) * 100:.0f}%"
                                                         for a, b in zip(derived_results, compare_results)]
            basic += ["Previous model size (of image)", "Overlap with previous model"]
        summary_df = summary_df.sort_values("Status", key=lambda col: col != "Check", kind="stable")
        head, toggle = st.columns([3, 1], vertical_alignment="center")
        head.markdown("**All scans**")
        technical = toggle.toggle("Show technical columns", key="show_technical")
        st.dataframe(summary_df if technical else summary_df[basic], width="stretch", hide_index=True)

if st.session_state.get("skipped_files"):
    st.markdown("**Skipped uploads**")
    st.caption("These files didn't look like ultrasound and were not segmented.")
    st.dataframe(st.session_state["skipped_files"], width="stretch", hide_index=True)

with st.container(key="technical_details"):
    with st.expander("Model & research information"):
        st.write(f"**Deployed: {MODEL_LABEL}**")
        st.write("Internal test Dice (266 held-out scans, BUS-BRA · BUSI · BUS-UCLM): **89.06%**. "
                 "This is not clinical accuracy.")
        st.caption("Three U-Net++ models (ImageNet ResNet-34 encoder) trained with Focal Tversky loss and "
                   "scanner-appearance augmentation on BUS-BRA, BUSI, BUS-UCLM and BrEaST. Each model averages the "
                   "image and its mirror; the app averages the three. Input: grayscale 256 × 256, no despeckling. "
                   "Strict safetensors loading with SHA-256 checks. The Check flag uses a rule calibrated on the "
                   "held-out test set; it highlights likely weak segmentations but will miss some and flag some good ones.")
        st.text(f"Repository: {MODEL_REPOSITORY}\nRevision: {MODEL_REVISION}\nExperiment branch: {MODEL_BRANCH}")
        st.write(f"**Comparison: {checkpoint_exp25.MODEL_LABEL}**")
        st.caption("Earlier AMFU-Net (4-level U-Net with DAG attention) on SRAD-despeckled, CLAHE-enhanced input. "
                   "Internal test Dice 82.76% on the same 266 scans. Kept so its masks can be compared with the "
                   "deployed model; it has no Check flag.")
        st.text(f"Repository: {checkpoint_exp25.MODEL_REPOSITORY}\nRevision: {checkpoint_exp25.MODEL_REVISION}")
        st.caption("ModalityScan checks original images before any preprocessing. Its acceptance policy and calibration "
                   "are fixed. Open-set rejection is unmeasured. No reflection padding is applied at inference.")

st.html('<footer class="g17-footer">G17 final-year project, breast ultrasound lesion segmentation. For research '
        'use only.<br>Scans are processed inside this app and are not sent to any image, animation or analytics '
        'service. Keep patient identifiers out of uploads.</footer>')
