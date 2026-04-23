"""
Gas Sensor MTL Analyzer — Streamlit Web App
ResNet + Multi-Head Attention + Multi-Task Learning model for gas sensing kinetics.
"""

import os
import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Gas Sensor MTL Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

from model_utils import (
    load_model_bundle,
    load_model_bundle_from_dir,
    bundle_to_zip,
    parse_curves_df,
    parse_conditions_df,
    run_predict,
    run_finetune,
    compute_metrics,
    TARGET_NAMES,
    TARGET_LABELS,
)

# ── Data-format example tables ───────────────────────────────────────────────

def _curves_format_guide(key_suffix: str = ""):
    """Render an expander explaining all accepted curves_df file formats."""
    with st.expander("📋 Accepted file formats for curves", expanded=False):
        st.markdown("Choose any of the four layouts below. "
                    "All formats are converted to the same internal structure automatically.")

        # ---- PKL ----
        st.markdown("#### PKL format")
        st.caption("A pickled `pd.DataFrame` — must contain at least these two columns.")
        ex_pkl = pd.DataFrame({
            "curve_id": [1, 2, 3],
            "response":  [
                "np.array([0.010, 0.020, …, 0.980])  ← 1-D array, length N",
                "np.array([0.000, 0.010, …, 0.950])",
                "np.array([0.005, 0.015, …, 0.960])",
            ],
        })
        st.dataframe(ex_pkl, hide_index=True, width='stretch')

        st.divider()

        # ---- CSV Type 1 ----
        st.markdown("#### CSV Type 1 — Wide  *(one row per curve)*")
        st.caption(
            "First column: **curve_id**.  "
            "Remaining columns: response value at each time step "
            "(column headers can be any label, e.g. `t_0`, `t_1`, …)."
        )
        ex1 = pd.DataFrame({
            "curve_id": [1, 2, 3],
            "t_0":      [0.010, 0.000, 0.005],
            "t_1":      [0.020, 0.010, 0.015],
            "t_2":      [0.050, 0.030, 0.040],
            "t_N":      [0.980, 0.950, 0.960],
        })
        st.dataframe(ex1, hide_index=True, width='stretch')
        st.caption("(columns t_3 … t_N-1 omitted for brevity)")

        st.divider()

        # ---- CSV Type 2a — single channel ----
        st.markdown("#### CSV Type 2 — Long, single curve  *(one row per time point)*")
        st.caption(
            "First column: **curve_id**.  "
            "Second column: **time**.  "
            "Third column: **response**.  "
            "Rows with the same curve_id form one curve. Multiple curve_ids → multiple curves."
        )
        ex2a = pd.DataFrame({
            "curve_id": [1, 1, 1, 2, 2, 2],
            "time":     [0.000, 0.005, 0.010, 0.000, 0.005, 0.010],
            "response": [0.010, 0.020, 0.050, 0.000, 0.015, 0.040],
        })
        st.dataframe(ex2a, hide_index=True, width='stretch')

        st.divider()

        # ---- CSV Type 2b — multiple curves ----
        st.markdown("#### CSV Type 2 — Long, multiple curves")
        st.caption(
            "First column: **time** (shared across all curves).  "
            "Each subsequent column **response_{N}** holds one curve, "
            "where **N matches curve_id** in the conditions file."
        )
        ex2b = pd.DataFrame({
            "time":       [0.000, 0.005, 0.010, 0.015, 0.020],
            "response_1": [0.010, 0.020, 0.050, 0.080, 0.120],
            "response_2": [0.000, 0.010, 0.030, 0.060, 0.100],
            "response_3": [0.005, 0.015, 0.040, 0.070, 0.110],
        })
        st.dataframe(ex2b, hide_index=True, width='stretch')
        st.caption("ℹ️  If curve lengths differ from the model's expected input length "
                   "they are resampled automatically.")


def _conditions_format_guide(key_suffix: str = ""):
    """Render an expander explaining accepted conditions file formats."""
    with st.expander("📋 Accepted file formats for conditions", expanded=False):
        st.markdown(
            "The conditions file maps each **curve_id** to its known parameter values. "
            "It is used for fine-tuning (required) and for true-vs-predicted evaluation (optional)."
        )

        # ---- PKL ----
        st.markdown("#### PKL format")
        st.caption("A pickled `pd.DataFrame`. Must contain `curve_id` and at least one target column.")
        ex_pkl = pd.DataFrame({
            "curve_id":   [1, 2, 3],
            "t_ads":      [0.321, 0.324, 0.323],
            "t_des":      [0.338, 0.332, 0.337],
            "t_stableD":  [0.206, 0.203, 0.207],
            "t_stableA":  [0.134, 0.140, 0.131],
            "k_ads":      [9.51, 9.59, 9.64],
            "k_des":      [11.48, 11.39, 11.39],
            "noise_level":[0.014, 0.013, 0.013],
        })
        st.dataframe(ex_pkl, hide_index=True, width='stretch')

        st.divider()

        # ---- CSV ----
        st.markdown("#### CSV format")
        st.caption(
            "Same columns as above, saved as a plain CSV file. "
            "Only `curve_id` and the seven target columns are required; "
            "extra columns (e.g. `class`, `total_duration`) are ignored."
        )
        ex_csv = pd.DataFrame({
            "curve_id":   [1, 2, 3],
            "t_ads":      [0.321, 0.324, 0.323],
            "t_des":      [0.338, 0.332, 0.337],
            "t_stableD":  [0.206, 0.203, 0.207],
            "t_stableA":  [0.134, 0.140, 0.131],
            "k_ads":      [9.51, 9.59, 9.64],
            "k_des":      [11.48, 11.39, 11.39],
            "noise_level":[0.014, 0.013, 0.013],
        })
        st.dataframe(ex_csv, hide_index=True, width='stretch')

        st.markdown("**Target columns**")
        target_info = pd.DataFrame({
            "Column":      ["t_ads", "t_des", "t_stableD", "t_stableA",
                            "k_ads", "k_des", "noise_level"],
            "Description": [TARGET_LABELS[n] for n in TARGET_NAMES],
            "Range":       ["0 – 1 (fraction)", "0 – 1", "0 – 1", "0 – 1",
                            "> 0 (rate constant)", "> 0", "≥ 0"],
        })
        st.dataframe(target_info, hide_index=True, width='stretch')


# ── Segment visualization helpers ────────────────────────────────────────────
_SEG_ORDER  = ['t_ads', 't_stableA', 't_des', 't_stableD']
_SEG_COLORS = {
    't_ads':     '#2ecc71',  # green  — end of adsorption
    't_stableA': '#3498db',  # blue   — end of stable-A
    't_des':     '#e67e22',  # orange — end of desorption
    't_stableD': '#e74c3c',  # red    — end of stable-D
}
_SEG_LABELS = {
    't_ads':     'End Adsorption (t_ads)',
    't_stableA': 'End Stable-A (t_stableA)',
    't_des':     'End Desorption (t_des)',
    't_stableD': 'End Stable-D (t_stableD)',
}
# Shading for each segment region
_SEG_FILL = {
    't_ads':     'rgba(46,204,113,0.08)',
    't_stableA': 'rgba(52,152,219,0.08)',
    't_des':     'rgba(230,126,34,0.08)',
    't_stableD': 'rgba(231,76,60,0.08)',
}


def _segment_boundaries(pred_row: pd.Series) -> dict:
    """
    Cumulative time positions of each segment end on the normalized [0, 1] axis.
    The four fractions t_ads, t_stableA, t_des, t_stableD sum to 1.
    """
    ta  = float(pred_row['t_ads'])
    tsa = float(pred_row['t_stableA'])
    td  = float(pred_row['t_des'])
    tsd = float(pred_row['t_stableD'])
    return {
        't_ads':     ta,
        't_stableA': ta + tsa,
        't_des':     ta + tsa + td,
        't_stableD': min(ta + tsa + td + tsd, 1.0),
    }


def _response_at(pos: float, response_arr: np.ndarray) -> float:
    """Interpolate response value at a normalized time position."""
    x = np.linspace(0, 1, len(response_arr))
    return float(np.interp(pos, x, response_arr))


def plot_curves_with_segments(curves_subset: pd.DataFrame,
                               pred_df: pd.DataFrame) -> go.Figure:
    """
    Overlay sensor curves (lines) with colored circle markers at each of the
    4 predicted segment boundaries, plus light background shading per region.
    """
    fig = go.Figure()

    # Accumulate marker coords per segment type (one legend entry each)
    mx = {s: [] for s in _SEG_ORDER}
    my = {s: [] for s in _SEG_ORDER}
    mt = {s: [] for s in _SEG_ORDER}

    for _, crow in curves_subset.iterrows():
        cid  = crow['curve_id']
        resp = np.asarray(crow['response'])
        xarr = np.linspace(0, 1, len(resp))

        # Curve line (thin, semi-transparent)
        fig.add_trace(go.Scatter(
            x=xarr, y=resp,
            mode='lines',
            line=dict(width=1.2, color='rgba(80,80,80,0.35)'),
            showlegend=False,
            hovertemplate=f'<b>ID {cid}</b><br>t=%{{x:.3f}}<br>y=%{{y:.4f}}<extra></extra>',
        ))

        # Collect marker positions for each segment
        pred_rows = pred_df[pred_df['curve_id'] == cid]
        if pred_rows.empty:
            continue
        bounds = _segment_boundaries(pred_rows.iloc[0])
        for seg, pos in bounds.items():
            r = _response_at(pos, resp)
            mx[seg].append(pos)
            my[seg].append(r)
            mt[seg].append(
                f'<b>ID {cid}</b><br>{_SEG_LABELS[seg]}<br>'
                f't = {pos:.3f}<br>y = {r:.4f}'
            )

    # Add one scatter trace per segment for grouped legend + markers
    for seg in _SEG_ORDER:
        if not mx[seg]:
            continue
        fig.add_trace(go.Scatter(
            x=mx[seg], y=my[seg],
            mode='markers',
            marker=dict(
                size=11,
                color=_SEG_COLORS[seg],
                symbol='circle',
                line=dict(width=1.5, color='white'),
            ),
            name=_SEG_LABELS[seg],
            text=mt[seg],
            hovertemplate='%{text}<extra></extra>',
        ))

    # Segment shading — use the FIRST curve's boundaries as representative bands
    first_pred = pred_df.iloc[0]
    first_bounds = _segment_boundaries(first_pred)
    region_edges = [0.0,
                    first_bounds['t_ads'],
                    first_bounds['t_stableA'],
                    first_bounds['t_des'],
                    first_bounds['t_stableD']]
    for seg, (x0, x1) in zip(_SEG_ORDER, zip(region_edges, region_edges[1:])):
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=_SEG_FILL[seg],
            line_width=0,
            layer='below',
        )

    fig.update_layout(
        xaxis_title='Normalized time (0 → 1)',
        yaxis_title='Response',
        height=520,
        margin=dict(t=70, b=40, l=50, r=20),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.06,
            xanchor='left',   x=0,
            font=dict(size=11),
        ),
    )
    return fig


# ── Paths ─────────────────────────────────────────────────────────────────────
_APP_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR  = os.path.join(_APP_DIR, "model")
DEFAULT_CURVES_PKL = os.path.join(_APP_DIR, "data", "curves_df_sample.pkl")

# ── Auto-load default model (cached so it only loads once) ────────────────────
@st.cache_resource(show_spinner="Loading pretrained model…")
def _load_default_bundle():
    return load_model_bundle_from_dir(DEFAULT_MODEL_DIR)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    .step-box {
        background: #f0f4ff;
        border-left: 4px solid #4A90E2;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 Gas Sensor MTL")
    st.caption("ResNet · Attention · Multi-Task Learning")
    st.divider()

    # ---- Model source ----
    st.header("Model")
    model_source = st.radio(
        "Source",
        ["Default (pretrained)", "Upload custom ZIP"],
        index=0,
        label_visibility="collapsed",
    )

    bundle = None

    if model_source == "Default (pretrained)":
        try:
            bundle = _load_default_bundle()
            st.success("✅ Pretrained model loaded")
            st.caption(f"`./model/`  |  type: `{bundle['model_type']}`")
        except Exception as e:
            st.error(f"Could not load default model:\n{e}")

    else:
        model_zip_file = st.file_uploader(
            "Upload model ZIP",
            type=["zip"],
            help="ZIP must contain best_model.keras + scaler .pkl files",
        )
        if model_zip_file:
            cache_key = f"custom_{model_zip_file.name}"
            if st.session_state.get("_custom_bundle_key") != cache_key:
                with st.spinner("Loading…"):
                    try:
                        b = load_model_bundle(model_zip_file.read())
                        st.session_state["_custom_bundle"] = b
                        st.session_state["_custom_bundle_key"] = cache_key
                    except Exception as e:
                        st.error(f"Failed: {e}")
            bundle = st.session_state.get("_custom_bundle")
            if bundle:
                st.success(f"✅ Custom model loaded  |  `{bundle['model_type']}`")

    use_log1p = True  # always apply inverse log1p for k values

    if bundle is None:
        st.warning("No model loaded.")

    st.divider()
    st.caption("Built with Streamlit · TensorFlow · Plotly")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_visualize, tab_retrain, tab_about = st.tabs(
    ["🔮 Predict", "📊 Visualize", "🔧 Retrain", "ℹ️ About"]
)


# ════════════════════════════════════════════════════════
# TAB 1 — PREDICT  (step-by-step)
# ════════════════════════════════════════════════════════
with tab_predict:
    st.header("Predict Kinetic Parameters")

    if bundle is None:
        st.warning("⬅️  Please load a model first (sidebar).")
        st.stop()

    # ── STEP 1 — Select data ─────────────────────────────────────────────────
    st.markdown('<div class="step-box"><b>Step 1 — Select curve data</b></div>',
                unsafe_allow_html=True)

    has_default = os.path.isfile(DEFAULT_CURVES_PKL)
    data_source = st.radio(
        "Data source",
        (["Default test data", "Upload my own file"] if has_default
         else ["Upload my own file"]),
        horizontal=True,
        label_visibility="collapsed",
    )

    curves_df = None

    if data_source == "Default test data":
        try:
            curves_df = pd.read_pickle(DEFAULT_CURVES_PKL)
            st.success(f"✅ Default test data loaded — {len(curves_df):,} curves")
        except Exception as e:
            st.error(f"Could not read default file: {e}")

    else:
        _curves_format_guide("predict")
        uploaded = st.file_uploader(
            "Upload curves (.pkl or .csv)",
            type=["pkl", "csv"],
            key="pred_upload",
        )
        if uploaded:
            try:
                curves_df = parse_curves_df(uploaded)
                st.success(f"✅ {len(curves_df):,} curves loaded from `{uploaded.name}`")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # ── STEP 2 — Preview ─────────────────────────────────────────────────────
    if curves_df is not None:
        st.markdown('<div class="step-box"><b>Step 2 — Preview curves</b></div>',
                    unsafe_allow_html=True)

        col_info, col_chart = st.columns([1, 2], gap="large")

        with col_info:
            st.dataframe(
                curves_df[["curve_id"]].assign(
                    length=curves_df["response"].apply(len),
                    min=curves_df["response"].apply(np.min).round(4),
                    max=curves_df["response"].apply(np.max).round(4),
                ).head(10),
                width='stretch',
            )

        with col_chart:
            n_prev = st.slider(
                "Curves to preview", 1, min(50, len(curves_df)),
                min(8, len(curves_df)), key="n_prev"
            )
            subset = (
                curves_df.sample(n=n_prev, random_state=42)
                if len(curves_df) > n_prev else curves_df
            )
            fig_prev = go.Figure()
            for _, row in subset.iterrows():
                fig_prev.add_trace(go.Scatter(
                    y=row["response"],
                    name=f"ID {row['curve_id']}",
                    mode="lines", line=dict(width=1.2),
                ))
            fig_prev.update_layout(
                xaxis_title="Time step", yaxis_title="Response",
                height=340, margin=dict(t=10, b=30),
                legend=dict(font=dict(size=9)),
            )
            st.plotly_chart(fig_prev, width='stretch')

        # ── STEP 3 — Run prediction ───────────────────────────────────────────
        st.markdown('<div class="step-box"><b>Step 3 — Run prediction</b></div>',
                    unsafe_allow_html=True)

        col_btn, col_note = st.columns([1, 3])
        with col_btn:
            run_btn = st.button("▶  Run Prediction", type="primary", width='stretch')
        with col_note:
            st.caption(f"Model type: `{bundle['model_type']}` · {len(curves_df):,} curves")

        if run_btn:
            with st.spinner("Running inference…"):
                try:
                    pred_df = run_predict(bundle, curves_df, use_log1p=use_log1p)
                    st.session_state["pred_df"] = pred_df
                    st.session_state["pred_curves_df"] = curves_df
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.exception(e)

    # ── STEP 4 — Results ─────────────────────────────────────────────────────
    if "pred_df" in st.session_state:
        pred_df       = st.session_state["pred_df"]
        pred_curves   = st.session_state.get("pred_curves_df")

        st.markdown('<div class="step-box"><b>Step 4 — Results</b></div>',
                    unsafe_allow_html=True)

        st.success(f"✅ Done — {len(pred_df):,} predictions")

        # ── Prediction table ──────────────────────────────────────────────────
        fmt = {c: "{:.4f}" for c in TARGET_NAMES}
        st.dataframe(pred_df.style.format(fmt), width='stretch')

        # ── Curve segment preview ─────────────────────────────────────────────
        st.subheader("Curve Segment Preview")
        st.caption(
            "Each curve is drawn as a thin gray line. "
            "Colored circles mark the predicted end of each segment on the "
            "normalized time axis [0 → 1]. "
            "Background shading uses the first curve's boundaries as a reference."
        )

        if pred_curves is not None:
            avail_ids   = set(pred_df['curve_id'].values)
            matched_df  = pred_curves[pred_curves['curve_id'].isin(avail_ids)]

            max_show = min(30, len(matched_df))
            col_sl, col_seed = st.columns([3, 1])
            with col_sl:
                n_seg = st.slider(
                    "Curves to show", 1, max_show,
                    min(5, max_show), key="n_seg_preview"
                )
            with col_seed:
                seg_seed = st.number_input("Seed", value=42, step=1, key="seg_seed")

            show_df = (
                matched_df.sample(n=n_seg, random_state=int(seg_seed))
                if len(matched_df) > n_seg else matched_df
            )

            fig_seg = plot_curves_with_segments(show_df, pred_df)
            st.plotly_chart(fig_seg, width='stretch')

            # Colour legend reminder
            leg_cols = st.columns(4)
            for col, seg in zip(leg_cols, _SEG_ORDER):
                col.markdown(
                    f'<div style="background:{_SEG_COLORS[seg]};'
                    f'border-radius:4px;padding:4px 8px;color:white;'
                    f'font-size:0.8rem;text-align:center">'
                    f'{_SEG_LABELS[seg]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Curve data not available for segment preview.")

        # ── Single-curve detail ───────────────────────────────────────────────
        with st.expander("Single-curve detail + time fraction breakdown"):
            cid_options = pred_df['curve_id'].tolist()
            sel_cid = st.selectbox("Select curve ID", cid_options, key="detail_cid")

            sel_pred = pred_df[pred_df['curve_id'] == sel_cid].iloc[0]
            t_cols   = ['t_ads', 't_stableA', 't_des', 't_stableD']

            col_pie, col_vals = st.columns([1, 1])
            with col_pie:
                fig_pie = px.pie(
                    names=[_SEG_LABELS[c] for c in t_cols],
                    values=[sel_pred[c] for c in t_cols],
                    color_discrete_sequence=[_SEG_COLORS[c] for c in t_cols],
                    title=f"Curve ID {int(sel_cid)} — time fractions",
                )
                fig_pie.update_traces(textinfo='label+percent')
                st.plotly_chart(fig_pie, width='stretch')

            with col_vals:
                rows = [
                    {"Parameter": n, "Predicted": f"{sel_pred[n]:.4f}",
                     "Description": TARGET_LABELS[n]}
                    for n in TARGET_NAMES
                ]
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

                # Show single-curve segment plot if raw data is available
                if pred_curves is not None:
                    cid_row = pred_curves[pred_curves['curve_id'] == sel_cid]
                    if not cid_row.empty:
                        resp1 = np.asarray(cid_row.iloc[0]['response'])
                        xarr1 = np.linspace(0, 1, len(resp1))
                        bounds1 = _segment_boundaries(sel_pred)

                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(
                            x=xarr1, y=resp1, mode='lines',
                            line=dict(color='#555', width=1.5),
                            showlegend=False,
                        ))
                        for seg, pos in bounds1.items():
                            r = _response_at(pos, resp1)
                            fig1.add_trace(go.Scatter(
                                x=[pos], y=[r], mode='markers',
                                marker=dict(size=13, color=_SEG_COLORS[seg],
                                            line=dict(width=2, color='white')),
                                name=_SEG_LABELS[seg],
                            ))
                        fig1.update_layout(
                            xaxis_title='Normalized time',
                            yaxis_title='Response',
                            height=300, margin=dict(t=10, b=30),
                            legend=dict(font=dict(size=9)),
                        )
                        st.plotly_chart(fig1, width='stretch')

        # ── Summary stats + download ──────────────────────────────────────────
        with st.expander("Summary statistics"):
            st.dataframe(
                pred_df[TARGET_NAMES].describe().T.style.format("{:.4f}"),
                width='stretch',
            )

        st.download_button(
            "⬇  Download Predictions (CSV)",
            data=pred_df.to_csv(index=False).encode(),
            file_name="gas_sensor_predictions.csv",
            mime="text/csv",
        )


# ════════════════════════════════════════════════════════
# TAB 2 — VISUALIZE
# ════════════════════════════════════════════════════════
with tab_visualize:
    st.header("Visualize Results")

    _curves_format_guide("viz")
    col_a, col_b = st.columns(2)
    with col_a:
        viz_curves_file = st.file_uploader(
            "Curves (.pkl or .csv)",
            type=["pkl", "csv"],
            key="viz_curves",
        )
    with col_b:
        viz_cond_file = st.file_uploader(
            "Conditions with true labels (.pkl or .csv)  [optional]",
            type=["pkl", "csv"],
            key="viz_cond",
        )

    viz_curves_df = None
    viz_cond_df   = None

    if viz_curves_file:
        try:
            viz_curves_df = parse_curves_df(viz_curves_file)
        except Exception as e:
            st.error(f"Curves error: {e}")
    elif os.path.isfile(DEFAULT_CURVES_PKL):
        # Fall back to default if nothing uploaded yet
        try:
            viz_curves_df = pd.read_pickle(DEFAULT_CURVES_PKL)
            st.caption("ℹ️  Showing default curves. Upload a file above to override.")
        except Exception:
            pass

    if viz_cond_file:
        try:
            viz_cond_df = parse_conditions_df(viz_cond_file)
        except Exception as e:
            st.error(f"Conditions error: {e}")

    viz_pred_df = st.session_state.get("pred_df")
    if viz_pred_df is not None:
        st.info("ℹ️  Using predictions from the Predict tab. Upload conditions to compare true vs predicted.")

    # ── Curve explorer ────────────────────────────────────────────────────────
    if viz_curves_df is not None:
        st.subheader("Sensor Curve Explorer")

        ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 2])
        with ctrl1:
            max_n  = min(100, len(viz_curves_df))
            n_plot = st.slider("Number of curves", 1, max_n, min(10, max_n), key="viz_n")
        with ctrl2:
            seed = st.number_input("Seed", value=0, step=1, key="viz_seed")
        with ctrl3:
            has_preds = viz_pred_df is not None
            show_segs = st.checkbox(
                "Show time segment markers",
                value=False,
                key="viz_show_segs",
                disabled=not has_preds,
                help="Requires predictions from the Predict tab." if not has_preds
                     else "Overlay predicted segment boundaries on each curve.",
            )

        sample_df = (
            viz_curves_df.sample(n=n_plot, random_state=int(seed))
            if len(viz_curves_df) > n_plot else viz_curves_df
        )

        if show_segs and has_preds:
            # Use the shared segment-plot function with colored markers
            avail_ids = set(viz_pred_df["curve_id"].values)
            plot_df   = sample_df[sample_df["curve_id"].isin(avail_ids)]
            if plot_df.empty:
                st.warning("None of the sampled curves have predictions. "
                           "Run prediction first with the same data.")
                show_segs = False

        if show_segs and has_preds and not sample_df.empty:
            fig_c = plot_curves_with_segments(plot_df, viz_pred_df)
        else:
            fig_c = go.Figure()
            for _, row in sample_df.iterrows():
                xarr = np.linspace(0, 1, len(row["response"]))
                fig_c.add_trace(go.Scatter(
                    x=xarr, y=row["response"],
                    name=f"ID {row['curve_id']}",
                    mode="lines", line=dict(width=1), opacity=0.7,
                ))
            fig_c.update_layout(
                xaxis_title="Normalized time", yaxis_title="Response",
                height=520, margin=dict(t=70, b=40, l=50, r=20),
                legend=dict(font=dict(size=9)),
            )

        st.plotly_chart(fig_c, width='stretch')

    # ── Prediction distributions ──────────────────────────────────────────────
    if viz_pred_df is not None:
        st.subheader("Prediction Distributions")
        fig_h = make_subplots(rows=2, cols=4,
                              subplot_titles=[TARGET_LABELS[n] for n in TARGET_NAMES] + [""])
        for i, name in enumerate(TARGET_NAMES):
            r, c = divmod(i, 4)
            fig_h.add_trace(
                go.Histogram(x=viz_pred_df[name], name=name, showlegend=False,
                             marker_color=px.colors.qualitative.Plotly[i]),
                row=r + 1, col=c + 1,
            )
        fig_h.update_layout(height=480, margin=dict(t=50))
        st.plotly_chart(fig_h, width='stretch')

    # ── True vs Predicted ─────────────────────────────────────────────────────
    if viz_pred_df is not None and viz_cond_df is not None:
        st.subheader("True vs Predicted")
        avail = [n for n in TARGET_NAMES if n in viz_cond_df.columns]
        if not avail:
            st.warning("Conditions file has no recognised target columns.")
        else:
            metrics_df = compute_metrics(viz_cond_df, viz_pred_df)
            st.dataframe(metrics_df.style.format({"R²": "{:.4f}", "RMSE": "{:.6f}"}),
                         width='stretch')

            merged_ev = pd.merge(
                viz_cond_df[["curve_id"] + avail].rename(
                    columns={n: f"true_{n}" for n in avail}),
                viz_pred_df.rename(columns={n: f"pred_{n}" for n in avail}),
                on="curve_id",
            )
            ncols = 4
            nrows = -(-len(avail) // ncols)
            fig_sc = make_subplots(rows=nrows, cols=ncols,
                                   subplot_titles=[TARGET_LABELS.get(n, n) for n in avail])
            for i, name in enumerate(avail):
                r, c = divmod(i, ncols)
                tc, pc = f"true_{name}", f"pred_{name}"
                mn = min(merged_ev[tc].min(), merged_ev[pc].min())
                mx = max(merged_ev[tc].max(), merged_ev[pc].max())
                fig_sc.add_trace(go.Scatter(
                    x=merged_ev[tc], y=merged_ev[pc], mode="markers",
                    marker=dict(size=4, opacity=0.45,
                                color=px.colors.qualitative.Plotly[i % 10]),
                    showlegend=False,
                ), row=r + 1, col=c + 1)
                fig_sc.add_trace(go.Scatter(
                    x=[mn, mx], y=[mn, mx], mode="lines",
                    line=dict(color="red", dash="dash"), showlegend=False,
                ), row=r + 1, col=c + 1)
            fig_sc.update_layout(height=600, margin=dict(t=60))
            fig_sc.update_xaxes(title_text="True")
            fig_sc.update_yaxes(title_text="Predicted")
            st.plotly_chart(fig_sc, width='stretch')

            with st.expander("Per-curve detail table"):
                st.dataframe(merged_ev, width='stretch')


# ════════════════════════════════════════════════════════
# TAB 3 — RETRAIN / FINE-TUNE
# ════════════════════════════════════════════════════════
with tab_retrain:
    st.header("Fine-tune from Pretrained Model")

    if bundle is None:
        st.warning("⬅️  Please load a model first.")
        st.stop()

    st.markdown(
        "Upload **your own labeled dataset** to fine-tune the pretrained model. "
        "The retrained model can be downloaded as a new ZIP."
    )

    col_ft1, col_ft2 = st.columns(2)
    with col_ft1:
        _curves_format_guide("retrain")
        ft_curves_file = st.file_uploader("Training curves (.pkl or .csv)",
                                          type=["pkl", "csv"], key="ft_curves")
    with col_ft2:
        _conditions_format_guide("retrain")
        ft_cond_file = st.file_uploader("Training conditions (.pkl or .csv)",
                                        type=["pkl", "csv"], key="ft_cond")

    ft_curves_df = None
    ft_cond_df   = None

    if ft_curves_file:
        try:
            ft_curves_df = parse_curves_df(ft_curves_file)
            st.success(f"✅ {len(ft_curves_df):,} training curves loaded.")
        except Exception as e:
            st.error(f"Curves error: {e}")

    if ft_cond_file:
        try:
            ft_cond_df = parse_conditions_df(ft_cond_file)
            st.success(f"✅ {len(ft_cond_df):,} condition rows loaded.")
        except Exception as e:
            st.error(f"Conditions error: {e}")

    st.divider()
    st.subheader("Training Configuration")

    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        ft_epochs = st.number_input("Max epochs",  min_value=1,    max_value=2000, value=100,  step=10)
        ft_batch  = st.number_input("Batch size",  min_value=8,    max_value=512,  value=32,   step=8)
    with cfg2:
        ft_lr     = st.number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.2e")
        ft_refit  = st.checkbox("Re-fit scalers on new data",
                                help="Recommended if your data distribution differs from the original training set.")
    with cfg3:
        ft_freeze = st.checkbox("Freeze backbone  (heads only)",
                                help="Only train output heads. Faster; useful for small datasets.")

    st.divider()

    if ft_curves_df is None or ft_cond_df is None:
        st.info("Upload both curves and conditions to enable fine-tuning.")
    else:
        matched = len(set(ft_curves_df["curve_id"]) & set(ft_cond_df["curve_id"]))
        st.metric("Matched training samples", f"{matched:,}")

        if st.button("🚀 Start Fine-tuning", type="primary", width='stretch'):
            prog   = st.progress(0, text="Initialising…")
            status = st.empty()
            chart  = st.empty()
            hist   = {"epoch": [], "loss": [], "val_loss": []}

            def _cb(epoch, total, logs):
                pct = min(int(epoch / total * 100), 100)
                prog.progress(pct, text=f"Epoch {epoch}/{total}")
                status.markdown(
                    f"`train loss: {logs.get('loss', 0):.4f}` &nbsp; "
                    f"`val loss: {logs.get('val_loss', 0):.4f}`"
                )
                hist["epoch"].append(epoch)
                hist["loss"].append(logs.get("loss", 0))
                hist["val_loss"].append(logs.get("val_loss", 0))
                if len(hist["epoch"]) > 1:
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=hist["epoch"], y=hist["loss"],    name="Train"))
                    fig_l.add_trace(go.Scatter(x=hist["epoch"], y=hist["val_loss"], name="Val"))
                    fig_l.update_layout(height=280, margin=dict(t=20),
                                        xaxis_title="Epoch", yaxis_title="Loss")
                    chart.plotly_chart(fig_l, width='stretch')

            try:
                new_bundle, _ = run_finetune(
                    bundle=bundle,
                    curves_df=ft_curves_df,
                    conditions_df=ft_cond_df,
                    target_names=TARGET_NAMES,
                    epochs=int(ft_epochs), lr=float(ft_lr),
                    batch_size=int(ft_batch),
                    refit_scalers=ft_refit,
                    freeze_backbone=ft_freeze,
                    progress_callback=_cb,
                )
                st.session_state["finetuned_bundle"] = new_bundle
                prog.progress(100, text="Done!")
                st.success(f"✅ Fine-tuning complete! ({len(hist['epoch'])} epochs)")
            except Exception as e:
                st.error(f"Fine-tuning failed: {e}")
                st.exception(e)

    if "finetuned_bundle" in st.session_state:
        st.divider()
        st.subheader("Download Fine-tuned Model")
        if st.button("📦 Package model"):
            with st.spinner("Saving…"):
                try:
                    zip_bytes = bundle_to_zip(st.session_state["finetuned_bundle"])
                    st.download_button(
                        "⬇  Download finetuned_model.zip",
                        data=zip_bytes,
                        file_name="finetuned_model.zip",
                        mime="application/zip",
                    )
                except Exception as e:
                    st.error(f"Packaging failed: {e}")


# ════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════
with tab_about:
    st.header("About This App")
    st.markdown("""
## Gas Sensor MTL Analyzer

A web interface for the **ResNet + Multi-Head Attention + Multi-Task Learning (MTL)** model
that extracts kinetic parameters from gas sensor response curves.

---

### Model Outputs (7 parameters)
| Parameter | Description | Notes |
|-----------|-------------|-------|
| `t_ads` | Adsorption time fraction | Softmax (sum = 1) |
| `t_des` | Desorption time fraction | Softmax |
| `t_stableD` | Stable desorption fraction | Softmax |
| `t_stableA` | Stable adsorption fraction | Softmax |
| `k_ads` | Adsorption rate constant | Linear head |
| `k_des` | Desorption rate constant | Linear head |
| `noise_level` | Signal noise level | Linear head |

---

### Workflow

**🔮 Predict**
1. Model is auto-loaded from `./model/` on startup.
2. Select **Default test data** or upload your own curves_df.
3. Preview curves, then click **Run Prediction**.
4. Download results as CSV.

**📊 Visualize**
- Explore sensor curves interactively.
- See prediction distributions (histogram per target).
- Upload conditions_df (true labels) to see True vs Predicted scatter plots + R²/RMSE.

**🔧 Retrain**
- Upload labeled training data and fine-tune the pretrained model.
- Configure learning rate, epochs, and whether to freeze the backbone.
- Download the retrained model as a ZIP.

---

### Data Formats

**curves_df** — `.pkl` (`pd.DataFrame`) or `.csv`:
- `curve_id` : integer
- `response` : 1-D numpy array (or space-separated string in CSV)

**conditions_df** — `.pkl` or `.csv`:
- `curve_id` + any of the 7 target columns

**Model ZIP** (for custom upload):
```
model.zip
├── best_model.keras
├── scaler_X.pkl
├── scaler_k.pkl          ← combined-k variant
│   OR scaler_k_ads.pkl + scaler_k_des.pkl  ← split-k variant
└── scaler_noise.pkl
```

---

### Deployment

```bash
# Local
pip install -r requirements.txt
streamlit run app.py
```

For public sharing → push to GitHub and deploy on **Streamlit Cloud** (share.streamlit.io).
See `DEPLOY.md` for step-by-step instructions.
""")
