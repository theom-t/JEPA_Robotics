import cv2
import time

video_path = "/home/tmainetucker/Repos/JEPA_Robotics/data/lerobot_so100/videos/observation.images.top/chunk-000/file-000.mp4"
cap = cv2.VideoCapture(video_path)

start = time.time()
count = 0
while True:
    ret, frame = cap.read()
    if not ret or count >= 500:
        break
    # frame = cv2.resize(frame, (256, 256))
    count += 1

elapsed = time.time() - start
print(f"Raw Decode Speed: {count / elapsed:.2f} FPS")

cap = cv2.VideoCapture(video_path)
start = time.time()
count = 0
while True:
    ret, frame = cap.read()
    if not ret or count >= 500:
        break
    frame = cv2.resize(frame, (256, 256))
    count += 1

elapsed = time.time() - start
print(f"Decode + Resize Speed: {count / elapsed:.2f} FPS")

cap = cv2.VideoCapture(video_path)
start = time.time()
count = 0
while True:
    ret, frame = cap.read()
    if not ret or count >= 500:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (256, 256))
    count += 1

elapsed = time.time() - start
print(f"Full pipeline Speed: {count / elapsed:.2f} FPS")
