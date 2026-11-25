from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from workspace_ui import ProcessingState, processing_html, header_html, overview_html
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
    "final": dict(label="G17 final · U-Net++ ×3 ensemble", dice="89.06%",
                  loader=lambda: checkpoint.load_checkpoint(), preprocess=preprocess_ultrasound),
    "exp25": dict(label="Exp-25 · AMFU-Net (SRAD + CLAHE)", dice="82.76%",
                  loader=lambda: checkpoint_exp25.load_exp25(), preprocess=preprocess_exp25),
}
MODEL_CHOICES = {"final": "G17 final (U-Net++ ×3, 89.06%)", "exp25": "Exp-25 AMFU-Net (SRAD + CLAHE, 82.76%)",
                 "compare": "Compare both"}


def clear_analysis():
    for key in ("raw_batch_data", "compare_batch_data", "analysis_models", "skipped_files",
                "input_check_error", "analysis_state"):
        st.session_state.pop(key, None)


@st.cache_resource(show_spinner=False)
def load_input_check():
    return load_modality_check()


st.set_page_config(page_title="G17 · Ultrasound Research Workspace", page_icon="assets/scan-line.svg", layout="wide")


@st.cache_resource(show_spinner=False)
def model_resource(key="final"):
    return ModelResource()


def load_model(key="final", _on_load=None):
    """Download each model once and reuse it across predictions."""
    return model_resource(key).get(MODELS[key]["loader"], _on_load)


def difference_overlay(original_gray, mask_a, mask_b):
    """Green = both models, magenta = final model only, cyan = Exp-25 only."""
    rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    a, b = mask_a > 0, mask_b > 0
    colour = np.zeros_like(rgb)
    colour[a & b] = (60, 220, 120)
    colour[a & ~b] = (235, 70, 200)
    colour[~a & b] = (60, 200, 235)
    any_mask = (a | b)[..., None]
    blended = np.where(any_mask, 0.45 * rgb + 0.55 * colour, rgb)
    return blended.clip(0, 255).astype(np.uint8)


def mask_dice(mask_a, mask_b):
    a, b = mask_a > 0, mask_b > 0
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * (a & b).sum() / total)


def create_overlay(original_gray: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Place a red lesion region and yellow boundary over the ultrasound."""
    original_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB)
    colour_mask = np.zeros_like(original_rgb, dtype=np.uint8)
    colour_mask[:, :, 0] = binary_mask
    overlay = cv2.addWeighted(original_rgb, 0.72, colour_mask, 0.28, 0)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(
        overlay, contours, contourIdx=-1, color=(255, 255, 0), thickness=2
    )
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

        summary_rows.append(
            {
                "Filename": item["filename"],
                "Dimensions": f"{w} × {h}",
                "Lesion Area (px)": f"{lesion_pixels:,}",
                "Lesion Area (%)": f"{area_pct:.2f}%",
                "Mean Mask Probability": f"{mean_conf * 100:.1f}%" if lesion_pixels > 0 else "N/A",
            }
            | ({"Model agreement": f"{item['signals']['agreement'] * 100:.0f}%",
                "Check": "Review" if item["review"] else "—"} if item.get("review") is not None else {})
        )

    return derived, pd.DataFrame(summary_rows)


st.html(f"<style>{(Path(__file__).parent / 'assets/workspace.css').read_text(encoding='utf-8')}</style>")
st.html(header_html())
st.html('<p class="g17-notice">Research demonstration only. The model assumes a lesion is present; '
        'it does not replace clinical assessment. Do not upload identifiable patient data.</p>')

with st.container(key="upload_workspace"):
    st.html('<div class="g17-section"><span>01</span><h2>Scan workspace</h2></div>')
    st.caption("Add breast ultrasound images to check their appearance and segment lesion regions.")
    model_choice = st.radio(
        "Segmentation model", options=list(MODEL_CHOICES), format_func=MODEL_CHOICES.get, horizontal=True,
        key="model_choice", on_change=clear_analysis,
        help="G17 final is the deployed model. Exp-25 is the earlier AMFU-Net. Compare runs both on the same scans.",
    )
    uploaded_files = st.file_uploader(
        "Upload breast ultrasound scans", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
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
                    st.text(f"{index + 1:02} · {file.name}")
        if len(uploaded_files) > 4:
            st.caption(f"Showing 4 previews of {len(uploaded_files)} uploads. All files will be checked.")
        run_button = st.button(
            f"Analyze {len(uploaded_files)} scan{'s' if len(uploaded_files) > 1 else ''}",
            type="primary", width="stretch",
        )
    else:
        run_button = False
        st.html('<div class="g17-guidance"><span>Upload</span><b>→</b><span>Check</span><b>→</b>'
                '<span>Segment</span><b>→</b><span>Inspect</span></div>')
    st.caption("Breast ultrasound only. The appearance check does not establish breast anatomy, "
               "lesion presence or diagnostic suitability. Unrelated images may pass; ultrasound may be rejected.")

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
    st.info("No images passed the ultrasound input check. No segmentation was run.")
if state is not None and state.accepted is not None:
    st.html(overview_html(state.accepted, state.skipped, len(st.session_state.get("raw_batch_data", []))))

with st.container(key="threshold_workspace"):
    st.html('<div class="g17-section"><span>02</span><h2>Inspect results</h2></div>')
    threshold = st.slider(
        "Mask probability threshold", min_value=.10, max_value=.90, value=.50, step=.05,
        key="threshold", help="Pixels at or above this model probability enter the mask.",
    )
    st.caption(f"Current threshold: {threshold:.2f}. Lower values include more pixels; higher values "
               "produce smaller masks. Changes reuse the saved probability maps. Pixel probabilities "
               "are model outputs, not clinical confidence.")

if "raw_batch_data" in st.session_state:
    derived_results, summary_df = prepared or compute_derived_results(st.session_state["raw_batch_data"], threshold)
    model_keys = st.session_state.get("analysis_models", ["final"])
    primary = MODELS[model_keys[0]]
    compare_results = None
    if "compare_batch_data" in st.session_state:
        compare_results, _ = compute_derived_results(st.session_state["compare_batch_data"], threshold)
    st.success(f"Successfully processed **{len(derived_results)}** scan(s) at threshold **{threshold:.2f}** with "
               f"**{primary['label']}**" + (f" and **{MODELS[model_keys[1]]['label']}**." if compare_results else "."))
    flagged = sum(bool(item.get("review")) for item in derived_results)
    if flagged:
        st.warning(f"{flagged} of {len(derived_results)} scan(s) flagged **Check** — see the Batch summary.")
    with st.container(key="inspection_workspace"):
        # Indices are identities: even identical filenames remain independently selectable.
        selected_idx = st.selectbox(
            "Selected scan", options=list(range(len(derived_results))),
            format_func=lambda index: f"{index + 1:02} · {derived_results[index]['filename']}",
        )
        item = derived_results[selected_idx]
        h, w = item["original_size"]
        if item.get("review"):
            st.warning("**Check this scan.** The ensemble's quality signals fall outside the range where it was "
                       "reliable on the held-out test set (low agreement between models, an empty mask or an "
                       "unusually uncertain boundary). Treat this segmentation with extra caution.")
        if compare_results:
            other = compare_results[selected_idx]
            st.markdown("##### Model comparison")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Mask area · {MODELS['final']['label'].split(' · ')[0]}", f"{item['area_pct']:.2f}%")
            c2.metric(f"Mask area · {MODELS['exp25']['label'].split(' · ')[0]}", f"{other['area_pct']:.2f}%",
                      delta=f"{other['area_pct'] - item['area_pct']:+.2f} pts", delta_color="off")
            c3.metric("Overlap between the two masks (Dice)", f"{mask_dice(item['mask'], other['mask']) * 100:.1f}%")
            left, middle, right = st.columns(3, gap="medium")
            with left:
                st.image(item["overlay"], caption=f"{MODELS['final']['label']} ({MODELS['final']['dice']} test Dice)",
                         width="stretch", alt="Final model overlay")
            with middle:
                st.image(other["overlay"], caption=f"{MODELS['exp25']['label']} ({MODELS['exp25']['dice']} test Dice)",
                         width="stretch", alt="Exp-25 overlay")
            with right:
                st.image(difference_overlay(item["original"], item["mask"], other["mask"]),
                         caption="Difference · green both · magenta final only · cyan Exp-25 only",
                         width="stretch", alt="Difference between the two model masks")
            st.caption("Both models see the same accepted scan; Exp-25 sees it after SRAD + CLAHE, the final model "
                       "sees the raw grayscale image. Test Dice values are from the same 266 held-out scans.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Image dimensions", f"{w} × {h}")
            m2.metric("Mask area", f"{item['area_pct']:.2f}%")
            m3.metric("Mean mask probability", f"{item['mean_conf'] * 100:.1f}%" if item["lesion_detected"] else "N/A")
            st.caption(f"{primary['label']} · {item['lesion_pixels']:,} pixels in the mask. "
                       "Mean probability is calculated within this mask.")
            comparison = st.radio("Comparison layout", ["Side by side", "Overlay only"], horizontal=True)
            if comparison == "Side by side":
                left, right = st.columns(2, gap="medium")
                with left:
                    st.image(item["original"], caption="Original ultrasound", width="stretch", alt="Original ultrasound scan")
                with right:
                    st.image(item["overlay"], caption="Overlay · red region / yellow boundary", width="stretch", alt="Segmentation overlay with red mask and yellow boundary")
            else:
                st.image(item["overlay"], caption="Overlay · red region / yellow boundary", width="stretch", alt="Full-width segmentation overlay")
        extra = [k for k in ("enhanced", "despeckled") if k in item]
        tabs = st.tabs(["Probability", "Mask", "Model input"] + [{"enhanced": "Enhanced", "despeckled": "SRAD"}[k] for k in extra])
        with tabs[0]:
            st.image(item["probability"], caption="Sigmoid probability map · 0.0 to 1.0", width="stretch", clamp=True, alt="Continuous lesion probability map")
        with tabs[1]:
            st.image(item["mask"], caption=f"Binary lesion mask · threshold {threshold:.2f}", width="stretch", alt="Binary lesion segmentation mask")
        with tabs[2]:
            if "enhanced" in item:
                st.image(item["model_input"], caption="256 × 256 model input after SRAD + CLAHE", width="stretch", alt="Model input image")
            else:
                st.image(item["model_input"], caption="Grayscale 256 × 256 model input (no despeckling or CLAHE)",
                         width="stretch", alt="Model input image")
                s = item["signals"]
                st.caption(f"Model agreement {s['agreement'] * 100:.0f}% · uncertain-boundary ratio {s['ambiguity']:.2f} · "
                           f"{s['components']} region(s). Agreement is the mean Dice between the three models' masks.")
        for tab, key in zip(tabs[3:], extra):
            with tab:
                caption = "SRAD + CLAHE enhanced ultrasound" if key == "enhanced" else "SRAD despeckled ultrasound"
                st.image(item[key], caption=caption, width="stretch", alt=caption)
    if compare_results:
        summary_df["Exp-25 area (%)"] = [f"{r['area_pct']:.2f}%" for r in compare_results]
        summary_df["Overlap with Exp-25 (Dice)"] = [f"{mask_dice(a['mask'], b['mask']) * 100:.1f}%"
                                                    for a, b in zip(derived_results, compare_results)]
    st.subheader("Batch summary")
    st.dataframe(summary_df, width="stretch", hide_index=True)
else:
    st.caption("Your scan comparison, masks and probability maps will appear here after analysis.")

if st.session_state.get("skipped_files"):
    st.subheader("Skipped files")
    st.caption("These files did not pass the ultrasound appearance check and were not segmented.")
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

st.html('<footer class="g17-footer">G17 · Ultrasound research workspace<br>'
        'Research use only. Keep patient identifiers out of uploads. Images are processed by this app; '
        'no external image, animation or analytics service receives them.</footer>')
