from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("animal_detection_output.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(
        frame,
        classes=animal_classes,
        conf=0.35,
        imgsz=640,
        verbose=False
    )

    annotated_frame = results[0].plot() if len(results[0].boxes) > 0 else frame

    out.write(annotated_frame)
    cv2.imshow("Animal Detection - Webcam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
