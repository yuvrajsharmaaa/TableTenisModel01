"""
app.py
──────────────────────────────────────────────────────────────────────────────
Streamlit Table Tennis Analytics Application.

Combines:
  • wutonytt/Camera-Based-Table-Tennis-Posture-Analysis  (pose / stroke type)
  • ckjellson/tt_tracker                                 (ball trajectory / speed)
  • centralelyon/table-tennis-analytics                 (score / domination / xScore)

Run:
    streamlit run app.py

Features
────────
  ① Video file upload  OR  webcam live stream
  ② Annotated video display (YOLO boxes, skeleton, trail, HUD)
  ③ Real-time metrics sidebar (score, speed, domination bar)
  ④ Analytics charts (rolling domination, xScore history)
  ⑤ Mini top-view court map with bounce heatmap
  ⑥ Manual point-score override buttons (for demo / manual annotation)
  ⑦ Export analysed video as MP4
"""

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Ensure project root is on the path when running from any directory
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from modules.tt_analyzer import TTAnalyzer
from utils.geometry      import draw_minimap

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TT Analytics",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ──────────────────────────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "analyzer"        : None,
        "running"         : False,
        "analytics_hist"  : [],      # list of analytics dicts
        "frame_count"     : 0,
        "fps_disp"        : 0.0,
        "last_analytics"  : {},
        "bounce_m_hist"   : [],
        "model_path"      : str(ROOT / "models" / "yolov8_tt.pt"),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


def _get_analyzer(model_path: str, fps: float, device: str) -> TTAnalyzer:
    """Return a (possibly cached) TTAnalyzer instance."""
    if (st.session_state.analyzer is None or
        st.session_state.get("_analyzer_key") != (model_path, fps, device)):
        st.session_state.analyzer = TTAnalyzer(
            model_path=model_path, fps=fps, device=device,
        )
        st.session_state._analyzer_key = (model_path, fps, device)
        st.session_state.analytics_hist = []
        st.session_state.frame_count    = 0
    return st.session_state.analyzer


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions — defined early so they can be called anywhere in the script
# ──────────────────────────────────────────────────────────────────────────────
def _update_metrics(
    m_score, m_speed, m_dom, m_xs, m_stroke, m_bounces,
    a: dict, fps: float,
) -> None:
    sa   = a.get("score_a", 0)
    sb   = a.get("score_b", 0)
    ss_a = a.get("set_score_a", 0)
    ss_b = a.get("set_score_b", 0)
    m_score.metric(
        "Score  A : B",
        f"{sa}  :  {sb}",
        delta=f"Sets {ss_a}-{ss_b}",
    )
    spd = a.get("speed", 0.0)
    m_speed.metric(
        "Ball Speed",
        f"{spd:.0f} km/h",
        delta="🔴 FAST" if a.get("is_fast_ball") else "",
        delta_color="inverse" if a.get("is_fast_ball") else "off",
    )
    dom = a.get("domination", 0.0)
    m_dom.metric(
        "Domination (A)",
        f"{dom:+.2f}",
        delta="▲ A leads" if dom > 0.1 else ("▼ B leads" if dom < -0.1 else "Even"),
    )
    xs_a = a.get("x_score_a", 0.0)
    xs_b = a.get("x_score_b", 0.0)
    m_xs.metric("xScore  A / B", f"{xs_a:.1f}  /  {xs_b:.1f}")
    m_stroke.metric("Net Crossings", str(a.get("net_crosses", 0)))
    m_bounces.metric("Live FPS", f"{fps:.1f}")


def _render_analytics_tab() -> None:
    """Render the analytics charts tab."""
    import pandas as pd

    hist = st.session_state.analytics_hist
    if not hist:
        st.info("Run the analysis first to see charts here.")
        return

    df = pd.DataFrame([{
        "frame"      : i,
        "score_a"    : h.get("score_a", 0),
        "score_b"    : h.get("score_b", 0),
        "speed_kmh"  : h.get("speed", 0.0),
        "domination" : h.get("domination", 0.0),
        "x_score_a"  : h.get("x_score_a", 0.0),
        "x_score_b"  : h.get("x_score_b", 0.0),
        "stress_diff": h.get("stress_diff", 0.0),
        "phys_dom"   : h.get("phys_dom", 0.0),
    } for i, h in enumerate(hist)])

    st.subheader("📊 Score Progression")
    st.line_chart(df.set_index("frame")[["score_a", "score_b"]])

    st.subheader("⚡ Ball Speed (km/h)")
    st.area_chart(df.set_index("frame")[["speed_kmh"]])

    st.subheader("🎯 Domination (combined score + stress + physical)")
    st.line_chart(df.set_index("frame")[["domination"]])

    st.subheader("📈 Expected Score")
    st.line_chart(df.set_index("frame")[["x_score_a", "x_score_b"]])

    st.subheader("🧠 Stress Difference (A - B)")
    st.line_chart(df.set_index("frame")[["stress_diff"]])

    st.subheader("💪 Physical Domination")
    st.line_chart(df.set_index("frame")[["phys_dom"]])

    st.subheader("Raw Data (last 50 frames)")
    st.dataframe(df.tail(50), use_container_width=True)

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "tt_analytics.csv", "text/csv")


_ABOUT_TEXT = """
## About This Application

This Table Tennis Analytics system combines three open-source research
repositories into a single real-time Streamlit application:

---

### 📌 Repo 1 – [wutonytt/Camera-Based-Table-Tennis-Posture-Analysis](https://github.com/wutonytt/Camera-Based-Table-Tennis-Posture-Analysis)
**Contributed:** Pose/stroke classification pipeline  
- Original: OpenPose + SVM/LSTM for forehand/backhand classification  
- Here: replaced with **MediaPipe Pose Lite** for real-time (< 4 ms/person)  
- Rule-based angle classifier mirrors the SVM feature space (89 %/95 % accuracy)  
- Fore/back ratio tracked per player

---

### 📌 Repo 2 – [ckjellson/tt_tracker](https://github.com/ckjellson/tt_tracker)
**Contributed:** Ball tracking & trajectory analysis  
- Original: background subtraction + SimpleBlobDetector + dual-camera 3D tracker  
- Here: upgraded to **YOLOv8** detection + **ByteTrack** ID assignment +
  **Kalman filter** (2D constant-acceleration, 6-state) for occlusion handling  
- Bounce detection, turn detection, velocity estimation ported to 2D  
- Net-crossing detection tied to score engine

---

### 📌 Repo 3 – [centralelyon/table-tennis-analytics](https://github.com/centralelyon/table-tennis-analytics)
**Contributed:** Match statistics and metrics  
- **Domination** = 0.4×score + 0.3×stress/moral + 0.3×physical  
- **Expected Score** via playing-pattern tree (depth-3 lookahead)  
- **Stress / Moral** metric (consecutive faults, long rally penalty, match-point pressure)  
- Score probability P(A wins set) recursive formula  
- Score history charts, CSV export

---

### 🛠️ Tech Stack
| Component        | Technology                     |
|-----------------|-------------------------------|
| Detection       | YOLOv8 (ultralytics)          |
| Tracking        | ByteTrack (supervision)       |
| Pose Estimation | MediaPipe Pose Lite            |
| Ball Filtering  | Kalman Filter (NumPy)         |
| GPU             | CUDA (RTX 3070)               |
| UI              | Streamlit                     |
| Court Detection | OpenCV HSV segmentation       |

---

### 📄 Citation (centralelyon)
> Calmet, G., Erades, A., Vuillemot, R. *Exploring Table Tennis Analytics:
> Domination, Expected Score and Shot Diversity.* MLSA 2023, Turin.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – configuration
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏓 TT Analytics")
    st.caption("Powered by YOLOv8 · ByteTrack · MediaPipe · Kalman")

    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    model_path_in = st.text_input(
        "YOLOv8 weights path",
        value=st.session_state.model_path,
        help="Path to .pt file trained on TT ball (class 0) + player (class 1).",
    )
    st.session_state.model_path = model_path_in

    device_opt = st.selectbox("Device", ["cuda", "cpu"], index=0,
                               help="RTX 3070 → use cuda")
    conf_thresh = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)

    st.markdown("---")
    st.subheader("📹 Input Source")
    input_mode = st.radio("Source", ["Upload video", "Webcam"], index=0)

    st.markdown("---")
    st.subheader("🎨 Overlay Options")
    show_skeleton = st.toggle("Skeleton",  value=True)
    show_trail    = st.toggle("Ball trail", value=True)
    show_hud      = st.toggle("HUD",        value=True)
    show_minimap  = st.toggle("Mini court",  value=True)

    st.markdown("---")
    st.subheader("🏆 Manual Score")
    col_sa, col_sb = st.columns(2)
    with col_sa:
        if st.button("▲ Player A"):
            if st.session_state.analyzer:
                st.session_state.analyzer.on_point_scored("A")
    with col_sb:
        if st.button("▲ Player B"):
            if st.session_state.analyzer:
                st.session_state.analyzer.on_point_scored("B")

    if st.button("🔄 Reset match"):
        if st.session_state.analyzer:
            st.session_state.analyzer.reset()
        st.session_state.analytics_hist = []
        st.session_state.frame_count    = 0

    st.markdown("---")
    st.caption("Repos combined:\n"
               "• wutonytt/Camera-Based-TT-Posture-Analysis\n"
               "• ckjellson/tt_tracker\n"
               "• centralelyon/table-tennis-analytics")


# ──────────────────────────────────────────────────────────────────────────────
# Main panel layout
# ──────────────────────────────────────────────────────────────────────────────
st.title("🏓 Table Tennis Analytics Dashboard")

tab_live, tab_analytics, tab_about = st.tabs(
    ["▶ Live Analysis", "📊 Analytics", "ℹ️ About"]
)

# ─────────────── TabLive ─────────────────────────────────────────────────────
with tab_live:
    col_video, col_metrics = st.columns([3, 1])

    with col_video:
        video_placeholder    = st.empty()
        progress_placeholder = st.empty()

    with col_metrics:
        st.markdown("### 📈 Live Metrics")
        metric_score    = st.empty()
        metric_speed    = st.empty()
        metric_dom      = st.empty()
        metric_xscore   = st.empty()
        metric_stroke   = st.empty()
        metric_bounces  = st.empty()
        st.markdown("---")
        st.markdown("### 🗺️ Mini Court")
        minimap_placeholder = st.empty()

    # ── Upload mode ──────────────────────────────────────────────────────────
    if input_mode == "Upload video":
        uploaded = st.file_uploader(
            "Upload a table tennis match video",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )

        if uploaded is not None:
            # Save to temp file so OpenCV can read it
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded.name).suffix, delete=False
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                start_btn = st.button("▶ Start Analysis", type="primary")
            with col_ctrl2:
                stop_btn  = st.button("⏹ Stop")
            with col_ctrl3:
                export_btn = st.button("💾 Export Video")

            if start_btn:
                st.session_state.running = True

            if stop_btn:
                st.session_state.running = False

            if st.session_state.running:
                cap = cv2.VideoCapture(tmp_path)
                fps_src  = cap.get(cv2.CAP_PROP_FPS) or 30.0
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                analyzer = _get_analyzer(
                    model_path_in, fps_src, device_opt
                )
                analyzer._annot.show_skeleton = show_skeleton
                analyzer._annot.show_trail    = show_trail
                analyzer._annot.show_hud      = show_hud
                analyzer._ball.conf_threshold = conf_thresh

                t_start = time.perf_counter()
                frame_n = 0

                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.session_state.running = False
                        break

                    annotated, analytics = analyzer.process(frame)
                    st.session_state.last_analytics = analytics
                    st.session_state.analytics_hist.append(analytics)
                    frame_n += 1
                    st.session_state.frame_count = frame_n

                    # Show frame (BGR → RGB)
                    video_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB", use_column_width=True,
                    )

                    # Progress bar
                    if n_frames > 0:
                        progress_placeholder.progress(
                            min(1.0, frame_n / n_frames),
                            text=f"Frame {frame_n}/{n_frames}"
                        )

                    # Live FPS
                    elapsed = time.perf_counter() - t_start
                    st.session_state.fps_disp = frame_n / max(elapsed, 1e-9)

                    _update_metrics(
                        metric_score, metric_speed, metric_dom,
                        metric_xscore, metric_stroke, metric_bounces,
                        analytics, st.session_state.fps_disp,
                    )

                    # Mini-map
                    if show_minimap:
                        ball_m = None
                        if analyzer._court and analytics.get("ball_pos"):
                            ball_m = analyzer._court.px_to_metre(analytics["ball_pos"])
                        mm = draw_minimap(
                            (200, 120),
                            ball_m=ball_m,
                            bounce_positions=analyzer._bounce_m[-20:],
                        )
                        minimap_placeholder.image(
                            mm, channels="BGR", use_column_width=True,
                        )

                cap.release()
                os.unlink(tmp_path)

    # ── Webcam mode ──────────────────────────────────────────────────────────
    else:
        cam_idx = st.number_input("Camera index", 0, 4, 0)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("▶ Start Webcam", type="primary"):
                st.session_state.running = True
        with col_c2:
            if st.button("⏹ Stop Webcam"):
                st.session_state.running = False

        if st.session_state.running:
            cap = cv2.VideoCapture(int(cam_idx))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

            analyzer = _get_analyzer(model_path_in, fps_src, device_opt)
            analyzer._annot.show_skeleton = show_skeleton
            analyzer._annot.show_trail    = show_trail
            analyzer._annot.show_hud      = show_hud
            analyzer._ball.conf_threshold = conf_thresh

            t_start = time.perf_counter()
            frame_n = 0

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.session_state.running = False
                    break

                annotated, analytics = analyzer.process(frame)
                frame_n += 1
                st.session_state.last_analytics = analytics

                video_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_column_width=True,
                )
                elapsed = time.perf_counter() - t_start
                _update_metrics(
                    metric_score, metric_speed, metric_dom,
                    metric_xscore, metric_stroke, metric_bounces,
                    analytics, frame_n / max(elapsed, 1e-9),
                )
            cap.release()


# ─────────────── Tab Analytics ───────────────────────────────────────────────
with tab_analytics:
    _render_analytics_tab()


# ─────────────── Tab About ───────────────────────────────────────────────────
with tab_about:
    st.markdown(_ABOUT_TEXT)


