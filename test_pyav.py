import av
import time
import cv2

video_path = "/home/tmainetucker/Repos/JEPA_Robotics/data/lerobot_so100/videos/observation.images.top/chunk-000/file-000.mp4"

start = time.time()
container = av.open(video_path)
count = 0
for frame in container.decode(video=0):
    if count >= 500: break
    img = frame.to_ndarray(format='rgb24')
    # Resize
    img = cv2.resize(img, (256, 256))
    count += 1

elapsed = time.time() - start
print(f"PyAV Speed: {count / elapsed:.2f} FPS")
