from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("best.onnx")

results = model.predict("Test_Folder/rd.jpg", conf=0.4)

annotated = results[0].plot()

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

print("Detections:")
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"  {model.names[cls_id]}: {conf:.2%} confidence")