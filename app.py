from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from modeling_amfu import AMFUNet
from preprocessing import preprocess_ultrasound


MODEL_REPOSITORY = "aee4/G17-AMFU-Net-Paper"

st.set_page_config(page_title="G17 AMFU-Net Demo", page_icon="🔬", layout="wide")


@st.cache_resource(show_spinner="Loading the AMFU-Net model...")
def load_model():
    """Download the model once and reuse it across predictions."""
    loaded_model = AMFUNet.from_pretrained(MODEL_REPOSITORY)
    loaded_model.to("cpu")
    loaded_model.eval()
    return loaded_model


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


def _preprocess_single(file):
    """Preprocess a single uploaded ultrasound file."""
    img = Image.open(file)
    processed = preprocess_ultrasound(img)
    processed["filename"] = file.name
    return processed


def run_batch_inference(uploaded_files, model):
    """Preprocess all images concurrently, execute a single batched forward pass, and store probability maps."""
    with ThreadPoolExecutor() as executor:
        processed_items = list(executor.map(_preprocess_single, uploaded_files))

    tensors = [item["tensor"].squeeze(0) for item in processed_items]
    batch_tensor = torch.stack(tensors, dim=0).to("cpu")

    with torch.inference_mode():
        output = model(pixel_values=batch_tensor)
        probabilities = torch.sigmoid(output["logits"])

    results = []
    for i, item in enumerate(processed_items):
        prob_map_256 = probabilities[i, 0].detach().cpu().numpy()
        h, w = item["original_size"]
        prob_map = cv2.resize(prob_map_256, (w, h), interpolation=cv2.INTER_LINEAR)
        results.append(
            {
                "filename": item["filename"],
                "original": item["original"],
                "enhanced": item["enhanced"],
                "despeckled": item["despeckled"],
                "probability": prob_map,
                "original_size": (h, w),
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
                "Mean Confidence": f"{mean_conf * 100:.1f}%" if lesion_pixels > 0 else "N/A",
            }
        )

    return derived, pd.DataFrame(summary_rows)


st.title("G17 AMFU-Net Breast Lesion Segmentation")
st.write(
    "Upload one or more breast ultrasound images containing suspected lesions. "
    "The model will analyze each image and generate a segmentation mask."
)
st.warning(
    " **Research Demonstration Only:** The model assumes a lesion is present. "
    "It does not replace clinical pathology or radiological assessment."
)

with st.sidebar:
    st.header("⚙️ Prediction Settings")
    threshold = st.slider(
        "Segmentation Probability Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability cut-off for classifying a pixel as lesion.",
    )
    st.caption(
        "• **0.30:** Highly sensitive (larger mask, potential false positives)\n\n"
        "• **0.50:** Paper default standard operating threshold\n\n"
        "• **0.70:** Conservative (smaller mask, high-confidence cores only)"
    )
    st.divider()
    st.markdown("### Model Details")
    st.caption(
        "- **Architecture:** AMFU-Net (4-level U-Net + DAG + MFEF)\n"
        "- **Checkpoint:** `aee4/G17-AMFU-Net-Paper`\n"
        "- **Input Resolution:** $256 \\times 256$"
    )

uploaded_files = st.file_uploader(
    "Upload Breast Ultrasound Scans (Supports Multi-Image Batch)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or multiple ultrasound images above to begin batch analysis.")
else:
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_button = st.button(
            f"Segment {len(uploaded_files)} Image{'s' if len(uploaded_files) > 1 else ''}",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        st.caption(f"Loaded **{len(uploaded_files)}** scan(s) ready for enhancement and segmentation.")

    if run_button:
        try:
            model = load_model()
            with st.spinner(f"Enhancing and segmenting {len(uploaded_files)} scan(s)..."):
                batch_data = run_batch_inference(uploaded_files, model)
            st.session_state["raw_batch_data"] = batch_data
        except Exception as err:
            st.error(f"Inference error: {err}")

if "raw_batch_data" in st.session_state:
    raw_batch = st.session_state["raw_batch_data"]
    derived_results, summary_df = compute_derived_results(raw_batch, threshold)

    st.success(
        f"Successfully processed **{len(derived_results)}** scan(s) at threshold **{threshold:.2f}**."
    )

    # Batch summary table
    st.subheader("Batch Summary Report")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Detailed Scan Inspection")

    # Image selector
    filenames = [item["filename"] for item in derived_results]
    selected_idx = 0
    if len(filenames) > 1:
        selected_file = st.selectbox(
            "Select scan to inspect in detail:",
            filenames,
            index=0,
        )
        selected_idx = filenames.index(selected_file)

    active_item = derived_results[selected_idx]

    # Metrics row for selected scan
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected File", active_item["filename"])
    m2.metric("Scan Resolution", f"{active_item['original_size'][1]} × {active_item['original_size'][0]}")
    m3.metric("Lesion Area", f"{active_item['lesion_pixels']:,} px ({active_item['area_pct']:.2f}%)")
    m4.metric(
        "Mean Lesion Confidence",
        f"{active_item['mean_conf']*100:.1f}%" if active_item["lesion_detected"] else "N/A",
    )

    # 4-Column Side-by-Side View
    st.markdown("#### Side-by-Side Pipeline Stages")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(
            active_item["original"],
            caption="1. Original Ultrasound",
            use_container_width=True,
            clamp=True,
        )
    with col2:
        st.image(
            active_item["enhanced"],
            caption="2. SRAD + CLAHE Enhanced",
            use_container_width=True,
            clamp=True,
        )
    with col3:
        st.image(
            active_item["mask"],
            caption="3. Predicted Lesion Mask",
            use_container_width=True,
            clamp=True,
        )
    with col4:
        st.image(
            active_item["overlay"],
            caption="4. Lesion Overlay (Red) & Boundary (Yellow)",
            use_container_width=True,
        )

    # Detailed Tabs
    with st.expander("🔬 View High-Resolution Individual Layers & Probability Maps"):
        tab_orig, tab_srad, tab_enh, tab_prob, tab_mask, tab_over = st.tabs(
            [
                "Original",
                "SRAD Despeckled",
                "CLAHE Enhanced",
                "Probability Heatmap",
                "Binary Mask",
                "Blended Overlay",
            ]
        )
        with tab_orig:
            st.image(active_item["original"], caption="Original B-Mode Scan", use_container_width=True)
        with tab_srad:
            st.image(active_item["despeckled"], caption="Speckle-Reduced via SRAD", use_container_width=True)
        with tab_enh:
            st.image(active_item["enhanced"], caption="Contrast-Enhanced via CLAHE", use_container_width=True)
        with tab_prob:
            st.image(
                active_item["probability"],
                caption="Continuous Sigmoid Probability Map (0.0 to 1.0)",
                clamp=True,
                use_container_width=True,
            )
        with tab_mask:
            st.image(
                active_item["mask"],
                caption=f"Thresholded Binary Mask (T = {threshold:.2f})",
                use_container_width=True,
            )
        with tab_over:
            st.image(
                active_item["overlay"],
                caption="Contour Delineation Overlay",
                use_container_width=True,
            )

st.divider()
st.caption(
    "G17 Research Project | Model: AMFU-Net | Do not Upload Identifiable Patient Data | For Research Demonstration Purposes Only"
)
