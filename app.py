import cv2
import numpy as np
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


def create_overlay(original_gray, binary_mask):
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


def predict(image, threshold):
    """Preprocess one image and generate its mask and visual overlay."""
    processed = preprocess_ultrasound(image)
    model = load_model()
    input_tensor = processed["tensor"].to("cpu")

    with torch.inference_mode():
        output = model(pixel_values=input_tensor)
        probability = torch.sigmoid(output["logits"])

    probability = probability[0, 0].detach().cpu().numpy()
    original_height, original_width = processed["original_size"]
    probability = cv2.resize(
        probability,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    binary_mask = (probability >= float(threshold)).astype(np.uint8) * 255
    overlay = create_overlay(processed["original"], binary_mask)

    return {
        "enhanced": Image.fromarray(processed["enhanced"]),
        "probability": probability,
        "mask": Image.fromarray(binary_mask),
        "overlay": Image.fromarray(overlay),
    }


st.title("G17 AMFU-Net Breast Lesion Segmentation")
st.write(
    "Upload a breast-ultrasound image containing a suspected lesion. "
    "The app enhances the image and outlines the region predicted by AMFU-Net."
)
st.warning(
    "Research demonstration only. The model assumes that a lesion is present. "
    "It does not diagnose cancer or replace assessment by a clinician."
)

with st.sidebar:
    st.header("Prediction settings")
    threshold = st.slider(
        "Segmentation threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "A lower threshold marks more pixels. A higher threshold produces "
            "a more conservative mask."
        ),
    )
    st.caption(
        "0.30: more possible lesion pixels\n\n"
        "0.50: standard evaluation threshold\n\n"
        "0.70: only higher-confidence pixels"
    )

uploaded_file = st.file_uploader(
    "Upload a breast-ultrasound image", type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("Upload an image above to begin.")
else:
    uploaded_image = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_image, caption="Uploaded ultrasound", width=500)

    if st.button("Segment lesion", type="primary", use_container_width=True):
        try:
            with st.spinner("Enhancing the image and predicting the lesion..."):
                results = predict(uploaded_image, threshold)
            st.session_state["results"] = results
            st.session_state["threshold"] = threshold
        except Exception as error:
            st.error(f"Prediction failed: {error}")

if "results" in st.session_state:
    results = st.session_state["results"]
    st.success(
        "Segmentation completed at threshold "
        f"{st.session_state['threshold']:.2f}."
    )
    enhanced_tab, probability_tab, mask_tab, overlay_tab = st.tabs(
        ["Enhanced image", "Probability map", "Lesion mask", "Lesion overlay"]
    )

    with enhanced_tab:
        st.image(
            results["enhanced"],
            caption="After SRAD despeckling and CLAHE enhancement",
            use_container_width=True,
        )
    with probability_tab:
        st.image(
            results["probability"],
            caption="Brighter pixels have a higher predicted lesion probability",
            clamp=True,
            use_container_width=True,
        )
    with mask_tab:
        st.image(
            results["mask"],
            caption="White: predicted lesion | Black: background",
            use_container_width=True,
        )
    with overlay_tab:
        st.image(
            results["overlay"],
            caption="Red: predicted region | Yellow: predicted boundary",
            use_container_width=True,
        )

st.divider()
st.caption(
    "G17 adaptation: conventional B-mode ultrasound → SRAD → CLAHE → "
    "AMFU-Net → lesion mask. Do not upload identifiable patient data."
)
