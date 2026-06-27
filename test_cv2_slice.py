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
    frame = cv2.resize(frame, (256, 256))
    frame_rgb = frame[:, :, ::-1]
    # We must ensure it's contiguous for C-array copying later
    # Actually wait, `np.ascontiguousarray` might take time
    frame_rgb = frame_rgb.copy()
    count += 1

elapsed = time.time() - start
print(f"Slice + Copy Speed: {count / elapsed:.2f} FPS")

