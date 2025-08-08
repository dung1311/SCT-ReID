import subprocess
import cv2
import numpy as np

url = "rtsp://student:student123@192.168.1.218:7001/e8e896c9-c01d-cbf9-5af3-a6a208ea5925"
cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",
    "-i", url,
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-an",
    "-c:v", "rawvideo",
    "-pix_fmt", "bgr24",
    "-f", "rawvideo", "-"
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

w, h = 1920, 1080  # thay theo độ phân giải camera

while True:
    raw = proc.stdout.read(w * h * 3)
    if not raw:
        break
    frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

proc.terminate()
cv2.destroyAllWindows()
