from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
import numpy as np
import cv2

TOPIC = "/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"

def callback(msg: Image):
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB_INT8 -> BGR for OpenCV
    cv2.imshow("Drone Camera Feed", frame)
    cv2.waitKey(1)

node = Node()
success = node.subscribe(Image, TOPIC, callback)
if not success:
    print("Subscription failed — check topic name")
else:
    print(f"Subscribed to {TOPIC}, press Ctrl+C to exit")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        cv2.destroyAllWindows()