# Gas Sensor MTL Analyzer — Deployment Guide

## Local Run

```bash
cd "Gas sensing/app"
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Deploy to Streamlit Cloud (Free, Public URL)

1. Push the `app/` folder to a **public GitHub repo**:
   ```
   repo/
   ├── app.py
   ├── model_utils.py
   └── requirements.txt
   ```

2. Go to https://share.streamlit.io → **New app**
3. Select your repo, set **Main file path** = `app.py`
4. Click **Deploy**

> Note: Streamlit Cloud has a 1 GB RAM limit.  
> Fine-tuning large models may time out — use Hugging Face Spaces (below) for that.

---

## Deploy to Hugging Face Spaces (GPU support)

1. Create a Space at https://huggingface.co/spaces
2. Select **Streamlit** SDK
3. Upload `app.py`, `model_utils.py`, `requirements.txt`
4. For GPU, select a paid GPU tier in Space settings

---

## Prepare Your Model ZIP

From your Colab/local training output folder, create a ZIP with:

```
model.zip
├── best_model.keras
├── scaler_X.pkl
├── scaler_k.pkl          ← for runs 2-2 / 2-4
│   OR
├── scaler_k_ads.pkl      ← for run 2-3
├── scaler_k_des.pkl
└── scaler_noise.pkl
```

```python
import zipfile, os

save_path = "/content/drive/MyDrive/.../result/Resnet_Attention_MTL_2-3"
out_zip   = "/content/model_2-3.zip"

with zipfile.ZipFile(out_zip, "w") as zf:
    for fname in ["best_model.keras", "scaler_X.pkl",
                  "scaler_k_ads.pkl", "scaler_k_des.pkl", "scaler_noise.pkl"]:
        zf.write(os.path.join(save_path, fname), fname)

print("Done:", out_zip)
```

---

## Prepare Your Data CSV

If you don't have pickle files, export curves as CSV:

```python
# Wide format — each time step is a column
import pandas as pd, numpy as np

response_matrix = np.stack(curves_df["response"].values)   # (N, T)
cols = [f"t_{i}" for i in range(response_matrix.shape[1])]
wide_df = pd.DataFrame(response_matrix, columns=cols)
wide_df.insert(0, "curve_id", curves_df["curve_id"].values)
wide_df.to_csv("curves_wide.csv", index=False)

# conditions
conditions_df.to_csv("conditions.csv", index=False)
```
