# G17 AMFU-Net Streamlit Demo

This Streamlit application demonstrates breast-lesion segmentation with the
G17 paper-based AMFU-Net model. It applies SRAD despeckling and CLAHE
enhancement before generating a probability map, binary mask and overlay.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and add these files to its root.
2. Go to https://share.streamlit.io and select **Create app**.
3. Choose the repository and branch.
4. Set the main file path to `app.py`.
5. Deploy the app.

The weights are downloaded from `aee4/G17-AMFU-Net-Paper` when the app starts.
The model is cached after loading, so it is not downloaded for every prediction.

Research demonstration only. This model assumes a lesion is present and is not
a clinical diagnostic tool.
