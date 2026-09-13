# 🛣️ Road Damage Detection System

A real-time road damage detection system built with YOLO11, Streamlit, and Supabase. It detects road surface damage (longitudinal cracks, transverse cracks, alligator cracks, and potholes) from images, videos, or a live webcam feed, tags detections with GPS location, stores reports in a cloud database, and provides a municipal dashboard to review and export submitted reports.

## Features

- **Image upload detection** — upload a photo, see annotated results side-by-side with the original
- **Video upload detection** — processes a video and displays live-optimized detection frames with an FPS counter
- **Live webcam detection** — real-time detection from your device's camera
- **GPS location capture** — automatically captures the reporter's location via the browser (with permission)
- **Cloud reporting** — detections are saved to a Supabase (PostgreSQL) database
- **Municipal dashboard** — a built-in admin view showing all reports, filterable by damage type and source, with a live map and CSV export
- **Email reporting** — generates a pre-filled email report (with detection details, confidence, and location) that the user can review and choose to send or discard
- **Portable** — runs as a browser-based web app, usable from both desktop and mobile

## Model

- **Architecture:** YOLO11n (nano) — chosen for fast CPU inference so the system doesn't require a GPU to run
- **Classes:** Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole
- **Dataset:** RDD2022 (Road Damage Dataset 2022), ~47,000 images across 6 countries
- **Format:** Exported to ONNX for faster CPU inference
- **Performance:** ~0.55 mAP50 overall, ~0.43 mAP50 on the Pothole class specifically; ~15 FPS on a standard laptop CPU

## Tech Stack

| Layer | Tool |
|---|---|
| Object detection | Ultralytics YOLO11 |
| Inference runtime | ONNX Runtime |
| Web app / UI | Streamlit |
| Cloud database | Supabase (PostgreSQL) |
| Location | Browser Geolocation API (via `streamlit-geolocation`) |
| Email | SMTP (Gmail) |

## Project Structure

```
├── app.py                  # Main Streamlit app (detection + dashboard views)
├── best.pt                 # Trained YOLO11n model weights (PyTorch)
├── best.onnx                # Exported ONNX model used for inference
├── export.py                # Script to export best.pt to ONNX
├── test_image.py             # Standalone script to test detection on a single image
├── webcam_test.py            # Standalone script to test live webcam detection
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── secrets.toml          # API keys and credentials (NOT committed to git)
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install manually:

```bash
pip install streamlit ultralytics opencv-python numpy pillow supabase streamlit-geolocation pandas onnx onnxruntime
```

### 4. Set up Supabase (cloud database)

1. Create a free project at [supabase.com](https://supabase.com)
2. In the Table Editor, create a table named `damage_report` with these columns:

   | Column | Type |
   |---|---|
   | `id` | `int8` (auto-increment / identity, primary key) |
   | `created_at` | `timestamptz` (default `now()`) |
   | `damage_type` | `text` |
   | `confidence` | `float8` |
   | `latitude` | `float8` |
   | `longitude` | `float8` |
   | `source` | `text` |

3. In the SQL Editor, run:

   ```sql
   ALTER TABLE damage_report ENABLE ROW LEVEL SECURITY;

   CREATE POLICY "Allow public inserts" ON damage_report
   FOR INSERT TO anon
   WITH CHECK (true);

   CREATE POLICY "Allow public reads" ON damage_report
   FOR SELECT TO anon
   USING (true);
   ```

4. Go to **Project Settings → API** and copy your **Project URL** and **anon public key**.

### 5. Set up email sending (optional — for the "email report" feature)

1. Enable 2-Step Verification on your Google account
2. Generate an **App Password** under Google Account → Security → App Passwords
3. Copy the 16-character password

### 6. Configure secrets

Create a file at `.streamlit/secrets.toml`:

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-public-key"
EMAIL_ADDRESS = "your.email@gmail.com"
EMAIL_APP_PASSWORD = "your-16-character-app-password"
```

**Do not commit this file.** It's already excluded via `.gitignore`.

### 7. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

1. **Detect damage:** choose "Upload Image/Video" or "Live Webcam" mode, adjust the confidence threshold if needed
2. **Allow location access** when prompted by your browser, to tag reports with GPS coordinates
3. **Report:** click "Report this damage" to save the detection to the cloud database, or use the email option to generate a report you can review and send to a municipality
4. **View the dashboard:** click "🏛️ View Municipal Dashboard" to see all submitted reports, filter them, view them on a map, or export as CSV

## Re-training the Model

The model was trained on the [RDD2022 dataset](https://www.kaggle.com/datasets/sreekaraditya/rdd2022-yolo-crackscan-v2) using Ultralytics YOLO11:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(
    data="path/to/data.yaml",
    epochs=60,
    imgsz=640,
    batch=32,
    patience=15
)
```

After training, export to ONNX for faster CPU inference:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx")
```

## Known Limitations

- Pothole detection accuracy (~0.43 mAP50) is lower than other damage classes, largely due to potholes being underrepresented in the source dataset relative to crack types
- Live webcam mode uses a simple checkbox-driven loop rather than true WebRTC streaming, so it works best on desktop browsers; mobile browser support for continuous live camera detection is limited without further work (`streamlit-webrtc` integration)
- There is no direct integration with any real municipality's official complaint system — the "municipal dashboard" and "email report" features are designed to be a functional stand-in that a real deployment could connect to an actual government system if institutional access were available

## License

This project uses the RDD2022 dataset (CC BY-SA 4.0) and Ultralytics YOLO11 (AGPL-3.0). Check license compatibility before commercial use.