import os
import io
import zipfile
import tempfile
import math
import inspect
import warnings

# Suppress TF C++ info/warning logs (must be set before importing tensorflow)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Suppress Python-level UserWarnings from Keras and sklearn
warnings.filterwarnings('ignore', category=UserWarning, module='keras')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# ── Keras version-compatibility patch ────────────────────────────────────────
# TF 2.21 / Keras 3.12: models saved with certain Keras 3.x sub-versions may
# embed 'quantization_config' (and other newer keys) in Dense / Conv1D configs.
# The fix: replace Layer.from_config (and patch every known built-in layer that
# might override it) so that unknown kwargs are silently dropped before the
# constructor is called.

_STRIP_KEYS = frozenset({"quantization_config", "lora_rank", "lora_alpha"})


def _make_compat_from_config():
    """Return a classmethod that strips unknown config keys before calling cls()."""
    @classmethod  # type: ignore[misc]
    def _from_config(cls, config):
        config = {k: v for k, v in config.items() if k not in _STRIP_KEYS}
        try:
            return cls(**config)
        except TypeError:
            # Last-resort: filter to params the constructor actually accepts
            sig = inspect.signature(cls.__init__)
            valid = set(sig.parameters.keys()) | {"name", "trainable", "dtype"}
            return cls(**{k: v for k, v in config.items() if k in valid})
    return _from_config


def _patch_keras_compat():
    try:
        import keras

        compat_fc = _make_compat_from_config()

        # Patch the base class — covers all layers that don't override from_config
        keras.layers.Layer.from_config = compat_fc

        # Explicitly patch common layers that may override from_config themselves
        _targets = [
            "Dense", "Conv1D", "Conv2D", "Embedding",
            "BatchNormalization", "LayerNormalization",
            "MultiHeadAttention", "Dropout", "Activation",
            "MaxPooling1D", "GlobalAveragePooling1D", "GlobalMaxPooling1D",
            "Add", "Multiply", "Softmax",
        ]
        for name in _targets:
            layer_cls = getattr(keras.layers, name, None)
            if layer_cls is not None:
                layer_cls.from_config = compat_fc

    except Exception:
        pass  # never crash on a best-effort patch


_patch_keras_compat()

TARGET_NAMES = ['t_ads', 't_des', 't_stableD', 't_stableA', 'k_ads', 'k_des', 'noise_level']

TARGET_LABELS = {
    't_ads':       'Adsorption time fraction',
    't_des':       'Desorption time fraction',
    't_stableD':   'Stable-D time fraction',
    't_stableA':   'Stable-A time fraction',
    'k_ads':       'Adsorption rate (k_ads)',
    'k_des':       'Desorption rate (k_des)',
    'noise_level': 'Noise level',
}


# ── Custom Keras layers ──────────────────────────────────────────────────────

class AddPositionalEmbedding(tf.keras.layers.Layer):
    def call(self, inputs):
        x, pos_embed = inputs
        return x + pos_embed

    def get_config(self):
        return super().get_config()


class TemporalAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(1)
        self.softmax     = tf.keras.layers.Softmax(axis=1)
        self.multiply    = tf.keras.layers.Multiply()

    def build(self, input_shape):
        self.score_dense.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        score    = self.score_dense(x)
        weights  = self.softmax(score)
        weighted = self.multiply([weights, x])
        return tf.keras.ops.sum(weighted, axis=1)

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "AddPositionalEmbedding": AddPositionalEmbedding,
    "TemporalAttentionPooling": TemporalAttentionPooling,
}


# ── Model bundle helpers ─────────────────────────────────────────────────────

def _detect_model_type(file_list):
    """Return 'split_k' if separate k_ads/k_des scalers exist, else 'combined_k'."""
    names = {os.path.basename(f) for f in file_list}
    if "scaler_k_ads.pkl" in names and "scaler_k_des.pkl" in names:
        return "split_k"
    return "combined_k"


def _find_file(all_files, name):
    for f in all_files:
        if os.path.basename(f) == name:
            return f
    raise FileNotFoundError(f"{name} not found in uploaded ZIP.")


def _walk_files(directory):
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            result.append(os.path.join(root, f))
    return result


def _load_keras_model(model_path: str) -> tf.keras.Model:
    """
    Load a .keras model with multiple fallback strategies for Keras version mismatches.
    TF 2.21 / Keras 3.x: prefer keras.models.load_model over tf.keras.models.load_model.
    """
    import keras

    attempts = [
        # 1. keras directly — most reliable in Keras 3.x
        lambda: keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS),
        # 2. tf.keras — compatibility shim
        lambda: tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS),
        # 3. keras with safe_mode disabled
        lambda: keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS,
                                        safe_mode=False),
    ]

    last_err = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as e:
            last_err = e

    raise last_err


def load_model_bundle_from_dir(model_dir: str) -> dict:
    """Load model + scalers directly from a local directory (no ZIP needed)."""
    all_files = _walk_files(model_dir)
    model_type = _detect_model_type(all_files)

    model_path = _find_file(all_files, "best_model.keras")
    model = _load_keras_model(model_path)

    bundle = {
        "model":        model,
        "scaler_X":     joblib.load(_find_file(all_files, "scaler_X.pkl")),
        "scaler_noise": joblib.load(_find_file(all_files, "scaler_noise.pkl")),
        "model_type":   model_type,
        "_tmp_dir":     model_dir,
    }

    if model_type == "split_k":
        bundle["scaler_k_ads"] = joblib.load(_find_file(all_files, "scaler_k_ads.pkl"))
        bundle["scaler_k_des"] = joblib.load(_find_file(all_files, "scaler_k_des.pkl"))
    else:
        bundle["scaler_k"] = joblib.load(_find_file(all_files, "scaler_k.pkl"))

    return bundle


def load_model_bundle(zip_bytes: bytes) -> dict:
    """
    Unpack ZIP and load model + scalers.

    Returns a dict with:
        model, scaler_X, scaler_noise, model_type
        + scaler_k            (combined_k variant)
        + scaler_k_ads/k_des  (split_k variant)
    """
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(tmp_dir)

    all_files = _walk_files(tmp_dir)
    model_type = _detect_model_type(all_files)

    model_path = _find_file(all_files, "best_model.keras")
    model = _load_keras_model(model_path)

    bundle = {
        "model":        model,
        "scaler_X":     joblib.load(_find_file(all_files, "scaler_X.pkl")),
        "scaler_noise": joblib.load(_find_file(all_files, "scaler_noise.pkl")),
        "model_type":   model_type,
        "_tmp_dir":     tmp_dir,
    }

    if model_type == "split_k":
        bundle["scaler_k_ads"] = joblib.load(_find_file(all_files, "scaler_k_ads.pkl"))
        bundle["scaler_k_des"] = joblib.load(_find_file(all_files, "scaler_k_des.pkl"))
    else:
        bundle["scaler_k"] = joblib.load(_find_file(all_files, "scaler_k.pkl"))

    return bundle


def bundle_to_zip(bundle: dict) -> bytes:
    """Save model + scalers to a ZIP and return raw bytes."""
    tmp_dir = tempfile.mkdtemp()

    bundle["model"].save(os.path.join(tmp_dir, "best_model.keras"))
    joblib.dump(bundle["scaler_X"],     os.path.join(tmp_dir, "scaler_X.pkl"))
    joblib.dump(bundle["scaler_noise"], os.path.join(tmp_dir, "scaler_noise.pkl"))

    if bundle["model_type"] == "split_k":
        joblib.dump(bundle["scaler_k_ads"], os.path.join(tmp_dir, "scaler_k_ads.pkl"))
        joblib.dump(bundle["scaler_k_des"], os.path.join(tmp_dir, "scaler_k_des.pkl"))
    else:
        joblib.dump(bundle["scaler_k"], os.path.join(tmp_dir, "scaler_k.pkl"))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(tmp_dir):
            zf.write(os.path.join(tmp_dir, f), f)

    return buf.getvalue()


# ── Data parsing ─────────────────────────────────────────────────────────────

def _detect_csv_format(df: pd.DataFrame) -> str:
    """
    Identify which CSV layout the user provided.

    Returns one of:
      'response_col'  — has a 'response' column (string of floats / list-like)
      'wide'          — CSV Type 1: curve_id | val_0 | val_1 | ... | val_N
      'long_single'   — CSV Type 2 single: curve_id | time | response  (grouped by curve_id)
      'long_single_anon' — CSV Type 2 single (legacy): time | response  (one curve, auto id=1)
      'long_multi'    — CSV Type 2 multi:  time | response_1 | response_2 | ...
    """
    col_names = [c.strip().lower() for c in df.columns]
    first_col = col_names[0]

    # curve_id | time | response  → long_single (grouped)
    if (first_col == "curve_id" and len(col_names) >= 3
            and col_names[1] == "time" and "response" in col_names):
        return "long_single"

    if first_col == "time":
        if "response" in col_names:
            return "long_single_anon"
        import re
        if any(re.fullmatch(r"response_\d+", c) for c in col_names):
            return "long_multi"

    if "response" in df.columns:
        return "response_col"

    return "wide"


def _resample_response(response: np.ndarray, target_n: int) -> np.ndarray:
    """Resample a 1-D response to target_n points via linear interpolation."""
    if len(response) == target_n:
        return response
    x_old = np.linspace(0.0, 1.0, len(response))
    x_new = np.linspace(0.0, 1.0, target_n)
    return np.interp(x_new, x_old, response)


def parse_curves_df(uploaded_file, n_points: int | None = None) -> pd.DataFrame:
    """
    Parse curves_df from an uploaded .pkl or .csv file.

    Accepted formats
    ----------------
    PKL  — pd.DataFrame with columns: curve_id (int), response (np.ndarray)

    CSV Type 1 — Wide (one row per curve):
        curve_id | t_0 | t_1 | ... | t_N
        Each cell is the intensity at that time step.

    CSV Type 2 single — Long, one curve (one row per time point):
        time | response

    CSV Type 2 multi — Long, many curves (one row per time point, one column per curve):
        time | response_1 | response_2 | ...
        Column name response_N → curve_id = N (matches conditions file).

    Parameters
    ----------
    n_points : int or None
        If set, all response arrays are resampled to this length.
        Use this to match the model's expected input size.
    """
    name = getattr(uploaded_file, "name", str(uploaded_file))

    # ── PKL ──────────────────────────────────────────────────────────────────
    if name.endswith(".pkl"):
        df = pd.read_pickle(uploaded_file)
        if "response" not in df.columns:
            raise ValueError("PKL file must have a 'response' column.")
        df = df[["curve_id", "response"]].copy()
        df["response"] = df["response"].apply(np.asarray)

    # ── CSV ───────────────────────────────────────────────────────────────────
    elif name.endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
        fmt = _detect_csv_format(raw)

        if fmt == "response_col":
            # response cell is a space/comma-separated string
            def _parse_str(s):
                if isinstance(s, str):
                    return np.fromstring(s.replace(",", " "), sep=" ")
                return np.asarray(s, dtype=float)
            raw["response"] = raw["response"].apply(_parse_str)
            df = raw[["curve_id", "response"]].copy()

        elif fmt == "wide":
            # CSV Type 1: curve_id | val_0 | val_1 | ... | val_N
            id_col    = raw.columns[0]
            val_cols  = raw.columns[1:].tolist()
            curve_ids = raw[id_col].values
            responses = raw[val_cols].values.astype(float)
            df = pd.DataFrame({
                "curve_id": curve_ids,
                "response": [row for row in responses],
            })
            df["response"] = df["response"].apply(np.asarray)

        elif fmt == "long_single":
            # CSV Type 2 single: curve_id | time | response  (one or more curves)
            id_col   = raw.columns[0]
            time_col = raw.columns[1]
            resp_col = [c for c in raw.columns if c.strip().lower() == "response"][0]
            sorted_raw = raw.sort_values([id_col, time_col])
            records = []
            for cid, grp in sorted_raw.groupby(id_col, sort=True):
                records.append({"curve_id": cid, "response": grp[resp_col].values.astype(float)})
            df = pd.DataFrame(records)
            df["response"] = df["response"].apply(np.asarray)

        elif fmt == "long_single_anon":
            # CSV Type 2 single (legacy): time | response  (one curve, auto id=1)
            time_col = raw.columns[0]
            resp_col = [c for c in raw.columns if c.strip().lower() == "response"][0]
            grp = raw.sort_values(time_col)
            signal = grp[resp_col].values.astype(float)
            df = pd.DataFrame({"curve_id": [1], "response": [signal]})
            df["response"] = df["response"].apply(np.asarray)

        else:  # long_multi — CSV Type 2 multi: time | response_1 | response_2 | ...
            import re as _re
            time_col  = raw.columns[0]
            resp_cols = sorted(
                [c for c in raw.columns if _re.fullmatch(r"response_\d+", c.strip().lower())],
                key=lambda c: int(_re.search(r"\d+", c).group()),
            )
            sorted_raw = raw.sort_values(time_col)

            records = []
            for col in resp_cols:
                cid    = int(_re.search(r"\d+", col).group())
                signal = sorted_raw[col].values.astype(float)
                records.append({"curve_id": cid, "response": signal})

            df = pd.DataFrame(records)
            df["response"] = df["response"].apply(np.asarray)

    else:
        raise ValueError("Unsupported file format. Please upload a .pkl or .csv file.")

    if "curve_id" not in df.columns or "response" not in df.columns:
        raise ValueError("Parsed result must contain 'curve_id' and 'response' columns.")

    # Optional resampling to match model input length
    if n_points is not None:
        df["response"] = df["response"].apply(lambda r: _resample_response(r, n_points))

    return df.reset_index(drop=True)


def parse_conditions_df(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name
    if name.endswith(".pkl"):
        return pd.read_pickle(uploaded_file)
    return pd.read_csv(uploaded_file)


# ── Inference ────────────────────────────────────────────────────────────────

def run_predict(bundle: dict, curves_df: pd.DataFrame, use_log1p: bool = True) -> pd.DataFrame:
    """Run inference and return a DataFrame with curve_id + 7 predicted targets."""
    model       = bundle["model"]
    scaler_X    = bundle["scaler_X"]
    scaler_noise = bundle["scaler_noise"]
    model_type  = bundle["model_type"]

    X = np.stack(curves_df["response"].values)

    # Auto-resample if curve length doesn't match what the scaler was fitted on
    expected_n = scaler_X.n_features_in_
    if X.shape[1] != expected_n:
        X = np.array([_resample_response(row, expected_n) for row in X])

    X_scaled = np.expand_dims(scaler_X.transform(X), -1)
    pred = model.predict(X_scaled, verbose=0)

    t_pred     = pred["time_head"]
    noise_pred = scaler_noise.inverse_transform(pred["noise_head"])

    if model_type == "split_k":
        k_ads_pred = np.expm1(bundle["scaler_k_ads"].inverse_transform(pred["k_ads_head"]))
        k_des_pred = np.expm1(bundle["scaler_k_des"].inverse_transform(pred["k_des_head"]))
        pred_array = np.concatenate([t_pred, k_ads_pred, k_des_pred, noise_pred], axis=1)
    else:
        raw_k = bundle["scaler_k"].inverse_transform(pred["k_head"])
        k_pred = np.expm1(raw_k) if use_log1p else raw_k
        pred_array = np.concatenate([t_pred, k_pred, noise_pred], axis=1)

    pred_df = pd.DataFrame(pred_array, columns=TARGET_NAMES)
    pred_df.insert(0, "curve_id", curves_df["curve_id"].values)
    return pred_df


# ── Fine-tuning ──────────────────────────────────────────────────────────────

class _ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, fn):
        super().__init__()
        self.total_epochs = total_epochs
        self.fn = fn

    def on_epoch_end(self, epoch, logs=None):
        if self.fn:
            self.fn(epoch + 1, self.total_epochs, logs or {})


def run_finetune(
    bundle: dict,
    curves_df: pd.DataFrame,
    conditions_df: pd.DataFrame,
    target_names: list,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    refit_scalers: bool = False,
    freeze_backbone: bool = False,
    progress_callback=None,
) -> tuple:
    """
    Fine-tune the pretrained model on new labeled data.

    Returns (new_bundle, history_dict).
    history_dict has keys 'epoch', 'loss', 'val_loss'.
    """
    model      = bundle["model"]
    model_type = bundle["model_type"]

    merged = pd.merge(curves_df, conditions_df, on="curve_id", how="inner")
    if len(merged) == 0:
        raise ValueError("No matching curve_ids between curves_df and conditions_df.")

    X     = np.stack(merged["response"].values)
    y_all = merged[target_names].values.astype(np.float32)

    y_time  = y_all[:, 0:4]
    y_k     = y_all[:, 4:6]
    y_noise = y_all[:, 6:7]

    # ---- Scalers ----
    scaler_X     = RobustScaler().fit(X)     if refit_scalers else bundle["scaler_X"]
    scaler_noise = RobustScaler().fit(y_noise) if refit_scalers else bundle["scaler_noise"]

    X_scaled       = np.expand_dims(scaler_X.transform(X), -1)
    y_noise_scaled = scaler_noise.transform(y_noise)

    if model_type == "split_k":
        y_k_ads = y_k[:, 0:1]
        y_k_des = y_k[:, 1:2]
        scaler_k_ads = RobustScaler().fit(np.log1p(y_k_ads)) if refit_scalers else bundle["scaler_k_ads"]
        scaler_k_des = RobustScaler().fit(np.log1p(y_k_des)) if refit_scalers else bundle["scaler_k_des"]
        y_k_ads_sc   = scaler_k_ads.transform(np.log1p(y_k_ads))
        y_k_des_sc   = scaler_k_des.transform(np.log1p(y_k_des))

        splits = train_test_split(
            X_scaled, y_time, y_k_ads_sc, y_k_des_sc, y_noise_scaled,
            test_size=0.2, random_state=42
        )
        X_tr, X_v, yt_tr, yt_v, yka_tr, yka_v, ykd_tr, ykd_v, yn_tr, yn_v = splits

        train_y = {"time_head": yt_tr, "k_ads_head": yka_tr, "k_des_head": ykd_tr, "noise_head": yn_tr}
        val_y   = {"time_head": yt_v,  "k_ads_head": yka_v,  "k_des_head": ykd_v,  "noise_head": yn_v}
        losses  = {
            "time_head":  "mae",
            "k_ads_head": tf.keras.losses.Huber(delta=1.0),
            "k_des_head": tf.keras.losses.Huber(delta=1.0),
            "noise_head": "mse",
        }
        loss_weights = {"time_head": 1.0, "k_ads_head": 3.0, "k_des_head": 3.0, "noise_head": 0.5}
        new_k_scalers = {"scaler_k_ads": scaler_k_ads, "scaler_k_des": scaler_k_des}

    else:
        scaler_k = RobustScaler().fit(np.log1p(y_k)) if refit_scalers else bundle["scaler_k"]
        y_k_sc   = scaler_k.transform(np.log1p(y_k))

        splits = train_test_split(
            X_scaled, y_time, y_k_sc, y_noise_scaled,
            test_size=0.2, random_state=42
        )
        X_tr, X_v, yt_tr, yt_v, yk_tr, yk_v, yn_tr, yn_v = splits

        train_y = {"time_head": yt_tr, "k_head": yk_tr, "noise_head": yn_tr}
        val_y   = {"time_head": yt_v,  "k_head": yk_v,  "noise_head": yn_v}
        losses  = {"time_head": "mae", "k_head": "mse", "noise_head": "mse"}
        loss_weights = {"time_head": 2.0, "k_head": 1.0, "noise_head": 0.5}
        new_k_scalers = {"scaler_k": scaler_k}

    # ---- Freeze backbone if requested ----
    if freeze_backbone:
        trainable_names = {"time_head", "k_head", "k_ads_head", "k_des_head", "noise_head"}
        for layer in model.layers:
            layer.trainable = layer.name in trainable_names
    else:
        for layer in model.layers:
            layer.trainable = True

    metrics_cfg = {k: tf.keras.metrics.MeanAbsoluteError(name="mae") for k in losses}
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics_cfg,
    )

    history_data = {"epoch": [], "loss": [], "val_loss": []}

    class _HistCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            history_data["epoch"].append(epoch + 1)
            history_data["loss"].append(logs.get("loss", 0))
            history_data["val_loss"].append(logs.get("val_loss", 0))
            if progress_callback:
                progress_callback(epoch + 1, epochs, logs)

    model.fit(
        X_tr, train_y,
        validation_data=(X_v, val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping("val_loss", patience=20, restore_best_weights=True),
            _HistCallback(),
        ],
        verbose=0,
    )

    new_bundle = {
        "model":        model,
        "scaler_X":     scaler_X,
        "scaler_noise": scaler_noise,
        "model_type":   model_type,
        **new_k_scalers,
    }

    return new_bundle, history_data


# ── Evaluation helpers ───────────────────────────────────────────────────────

def compute_metrics(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute R² and RMSE for each target. Both DataFrames must share curve_id."""
    merged = pd.merge(
        true_df[["curve_id"] + TARGET_NAMES].rename(columns={n: f"true_{n}" for n in TARGET_NAMES}),
        pred_df.rename(columns={n: f"pred_{n}" for n in TARGET_NAMES}),
        on="curve_id",
    )
    rows = []
    for name in TARGET_NAMES:
        t = merged[f"true_{name}"]
        p = merged[f"pred_{name}"]
        rows.append({
            "Target":      name,
            "Description": TARGET_LABELS[name],
            "R²":          round(r2_score(t, p), 4),
            "RMSE":        round(math.sqrt(mean_squared_error(t, p)), 6),
        })
    return pd.DataFrame(rows)
