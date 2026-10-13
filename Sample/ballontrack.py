from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
import numpy as np
import cv2

TOPIC = "/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"

def detect_balloon(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around in HSV, so two ranges
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(largest)

    if radius < 5:  # filter noise
        return None

    return (int(x), int(y), int(radius))

def callback(msg: Image):
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    result = detect_balloon(frame)
    if result:
        x, y, radius = result
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        offset_x = x - center_x
        offset_y = y - center_y
        print(f"Balloon at ({x},{y}), radius={radius}, offset=({offset_x},{offset_y})")

    cv2.imshow("Drone Camera Feed", frame)
    cv2.waitKey(1)

node = Node()
success = node.subscribe(Image, TOPIC, callback)
if not success:
    print("Subscription failed")
else:
    print("Subscribed — press Ctrl+C to exit")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        cv2.destroyAllWindows()