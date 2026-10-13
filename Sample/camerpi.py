from picamera2 import Picamera2
import cv2
import numpy as np
import time

# ---------------- CONFIG ----------------
MIN_RADIUS_TO_TRACK = 5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------------- CAMERA SETUP ----------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)  # let auto-exposure/focus settle

# ---------------- DETECTION ----------------
def detect_balloon(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)

    if radius < MIN_RADIUS_TO_TRACK:
        return None, mask

    return (int(x), int(y), int(radius)), mask

# ---------------- MAIN LOOP ----------------
print("Starting detection - press 'q' to quit")

try:
    while True:
        frame = picam2.capture_array()  # comes as RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert for OpenCV display

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        result, mask = detect_balloon(frame)

        if result:
            x, y, radius = result
            offset_x = x - center_x
            offset_y = y - center_y
            print(f"Balloon: pos=({x},{y}) radius={radius} offset=({offset_x},{offset_y})")

            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 1)
        cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 1)

        cv2.imshow("Balloon Detection", frame)
        cv2.imshow("Mask (debug)", mask)  # helpful for tuning HSV values

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()