import cv2
from ultralytics import YOLO
import time

model = YOLO("best.onnx")

source = "Test_Folder/damage_road.mp4"  # 0 = webcam; or replace with a video file path like "test_video.mp4"

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Error: could not open video source")
    exit()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Road Damage Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()