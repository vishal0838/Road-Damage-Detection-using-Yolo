import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
from streamlit_geolocation import streamlit_geolocation

st.set_page_config(page_title="Road Damage Detection", layout="wide")

# ---------------------------------------------------------
# SETUP: Model + Supabase
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("best.onnx")

model = load_model()


@st.cache_resource
def init_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()


def report_damage(damage_type, confidence, latitude=None, longitude=None, source="unknown"):
    data = {
        "damage_type": damage_type,
        "confidence": confidence,
        "latitude": latitude,
        "longitude": longitude,
        "source": source
    }
    try:
        supabase.table("damage_report").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Failed to report: {e}")
        return False


def report_multiple(detections, source, latitude=None, longitude=None):
    success_count = 0
    for class_name, conf in detections:
        if report_damage(class_name, conf, latitude, longitude, source):
            success_count += 1
    return success_count


def fetch_reports():
    response = supabase.table("damage_report").select("*").order("created_at", desc=True).execute()
    return pd.DataFrame(response.data)


# ---------------------------------------------------------
# VIEW STATE — toggle between citizen app and municipal dashboard
# ---------------------------------------------------------
if "view" not in st.session_state:
    st.session_state.view = "citizen"

header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("🛣️ Road Damage Detection System")
with header_col2:
    st.write("")  # small vertical spacer
    if st.session_state.view == "citizen":
        if st.button("🏛️ View Municipal Dashboard"):
            st.session_state.view = "dashboard"
            st.rerun()
    else:
        if st.button("⬅️ Back to Detection App"):
            st.session_state.view = "citizen"
            st.rerun()

st.divider()

# ===========================================================
# VIEW 1: CITIZEN DETECTION APP
# ===========================================================
if st.session_state.view == "citizen":

    st.write("Upload an image or video, or use your live webcam, to detect road damage (cracks, potholes, etc.)")

    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05, key="conf_slider")

    # Real GPS location capture via browser
    st.subheader("📍 Location")
    location = streamlit_geolocation()

    manual_lat = None
    manual_lon = None

    if location and location.get("latitude") is not None:
        manual_lat = location["latitude"]
        manual_lon = location["longitude"]
        st.success(f"Location captured: {manual_lat:.6f}, {manual_lon:.6f}")
    else:
        st.info("Click the location icon above to share your location, or leave blank to report without location.")

    mode = st.radio("Choose input mode:", ["Upload Image/Video", "Live Webcam"], key="mode_selector")

    # ---- MODE 1: UPLOAD IMAGE OR VIDEO ----
    if mode == "Upload Image/Video":
        uploaded_file = st.file_uploader(
            "Upload an image or video",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
            key="file_uploader"
        )

        if uploaded_file is not None:
            file_type = uploaded_file.type

            # ---- IMAGE HANDLING ----
            if "image" in file_type:
                image = Image.open(uploaded_file)
                img_array = np.array(image)

                with st.spinner("Detecting damage..."):
                    results = model.predict(img_array, conf=conf_threshold)
                    annotated = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.image(image, use_container_width=True)
                with col2:
                    st.subheader("Detected Damage")
                    st.image(annotated_rgb, use_container_width=True)

                st.subheader("Detections")
                detections = []
                if len(results[0].boxes) == 0:
                    st.info("No damage detected above this confidence threshold.")
                else:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls_id]
                        detections.append((class_name, conf))
                        st.write(f"- **{class_name}**: {conf:.1%} confidence")

                    if st.button("📤 Report this damage", key="report_image_btn"):
                        count = report_multiple(detections, "image_upload", manual_lat, manual_lon)
                        st.success(f"Reported {count} damage detection(s) successfully!")

            # ---- VIDEO HANDLING ----
            elif "video" in file_type:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                tfile.close()

                st.subheader("Processing video...")
                stframe = st.empty()
                progress_bar = st.progress(0)
                fps_text = st.empty()

                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0

                display_width = 640
                infer_every_n = 2
                last_annotated = None
                prev_time = time.time()

                video_detections = {}

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    h, w = frame.shape[:2]
                    scale = display_width / w
                    frame_small = cv2.resize(frame, (display_width, int(h * scale)))

                    if frame_count % infer_every_n == 0:
                        results = model.predict(frame_small, conf=conf_threshold, verbose=False)
                        last_annotated = results[0].plot()

                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = model.names[cls_id]
                            if class_name not in video_detections or conf > video_detections[class_name]:
                                video_detections[class_name] = conf

                    annotated_rgb = (
                        cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB)
                        if last_annotated is not None
                        else cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    )

                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if prev_time else 0
                    prev_time = curr_time

                    stframe.image(annotated_rgb, channels="RGB", width=display_width)
                    fps_text.write(f"**Live FPS:** {fps:.1f}")

                    frame_count += 1
                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()

                try:
                    os.unlink(video_path)
                except PermissionError:
                    pass

                st.success("Video processing complete!")

                if video_detections:
                    st.subheader("Damage found in this video")
                    for class_name, conf in video_detections.items():
                        st.write(f"- **{class_name}**: up to {conf:.1%} confidence")

                    if st.button("📤 Report damage from this video", key="report_video_btn"):
                        detections_list = list(video_detections.items())
                        count = report_multiple(detections_list, "video_upload", manual_lat, manual_lon)
                        st.success(f"Reported {count} damage detection(s) successfully!")
                else:
                    st.info("No damage detected above this confidence threshold in this video.")

    # ---- MODE 2: LIVE WEBCAM ----
    elif mode == "Live Webcam":
        st.write("Check the box below to start live detection from your webcam. Uncheck to stop.")

        run_webcam = st.checkbox("Start Webcam", key="webcam_checkbox")
        stframe = st.empty()
        fps_text = st.empty()

        last_detections = []

        if run_webcam:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Could not access webcam. Check that no other app is using it, and that permissions are granted.")
            else:
                prev_time = 0

                while st.session_state.get("webcam_checkbox", True):
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam.")
                        break

                    results = model.predict(frame, conf=conf_threshold, verbose=False)
                    annotated_frame = results[0].plot()
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    last_detections = []
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        last_detections.append((model.names[cls_id], conf))

                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if prev_time else 0
                    prev_time = curr_time

                    stframe.image(annotated_rgb, channels="RGB", use_container_width=True)
                    fps_text.write(f"**FPS:** {fps:.1f}")

                cap.release()

            if last_detections:
                st.subheader("Last frame's detections")
                for class_name, conf in last_detections:
                    st.write(f"- **{class_name}**: {conf:.1%} confidence")

                if st.button("📤 Report last detected damage", key="report_webcam_btn"):
                    count = report_multiple(last_detections, "webcam", manual_lat, manual_lon)
                    st.success(f"Reported {count} damage detection(s) successfully!")

# ===========================================================
# VIEW 2: MUNICIPAL DASHBOARD
# ===========================================================
elif st.session_state.view == "dashboard":

    st.header("🏛️ Municipal Road Damage Dashboard")
    st.write("Live view of all citizen-reported road damage detections.")

    df = fetch_reports()

    if df.empty:
        st.info("No reports submitted yet.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reports", len(df))
        with col2:
            st.metric("Unique Damage Types", df["damage_type"].nunique())
        with col3:
            reports_with_location = df.dropna(subset=["latitude", "longitude"])
            st.metric("Reports with GPS", len(reports_with_location))

        st.divider()

        st.subheader("Filter Reports")
        col_a, col_b = st.columns(2)
        with col_a:
            damage_types = ["All"] + sorted(df["damage_type"].dropna().unique().tolist())
            selected_type = st.selectbox("Damage Type", damage_types)
        with col_b:
            sources = ["All"] + sorted(df["source"].dropna().unique().tolist())
            selected_source = st.selectbox("Source", sources)

        filtered_df = df.copy()
        if selected_type != "All":
            filtered_df = filtered_df[filtered_df["damage_type"] == selected_type]
        if selected_source != "All":
            filtered_df = filtered_df[filtered_df["source"] == selected_source]

        st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** reports")

        st.subheader("📍 Report Locations")
        map_df = filtered_df.dropna(subset=["latitude", "longitude"])
        if not map_df.empty:
            st.map(map_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
        else:
            st.info("No reports with location data to display on the map.")

        st.subheader("📋 Report Details")
        display_df = filtered_df[["id", "created_at", "damage_type", "confidence", "latitude", "longitude", "source"]].copy()
        display_df["confidence"] = (display_df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("📤 Export")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"road_damage_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )