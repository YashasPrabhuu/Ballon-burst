import asyncio
import threading
import numpy as np
import cv2

from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# ---------------- CONFIG ----------------
CAMERA_TOPIC = "/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"

FRAME_CENTER_DEADZONE = 20      # px, ignore small errors
MAX_YAW_RATE = 30.0             # deg/s
MAX_FORWARD_SPEED = 2.0         # m/s
MAX_VERTICAL_SPEED = 1.0        # m/s

TARGET_RADIUS_PX = 80           # desired apparent radius when "close enough" -> stop
MIN_RADIUS_TO_TRACK = 5         # ignore tiny/noisy detections

LOST_TARGET_TIMEOUT = 1.5       # seconds with no detection before we hold position

# Gains (tune these)
KP_YAW = 0.08        # deg/s per pixel of horizontal offset
KP_VERTICAL = 0.005  # m/s per pixel of vertical offset
KP_FORWARD = 0.02    # m/s per pixel of radius error

# ---------------- SHARED STATE ----------------
class TargetState:
    def __init__(self):
        self.lock = threading.Lock()
        self.offset_x = 0
        self.offset_y = 0
        self.radius = 0
        self.detected = False
        self.last_seen = 0.0

    def update(self, offset_x, offset_y, radius):
        with self.lock:
            self.offset_x = offset_x
            self.offset_y = offset_y
            self.radius = radius
            self.detected = True
            self.last_seen = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0

    def clear(self):
        with self.lock:
            self.detected = False

    def get(self):
        with self.lock:
            return self.offset_x, self.offset_y, self.radius, self.detected


target_state = TargetState()


# ---------------- VISION: runs in its own thread ----------------
def detect_balloon(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    if radius < MIN_RADIUS_TO_TRACK:
        return None

    return (int(x), int(y), int(radius))


def camera_callback(msg: Image):
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    result = detect_balloon(frame)

    if result:
        x, y, radius = result
        offset_x = x - center_x
        offset_y = y - center_y
        target_state.update(offset_x, offset_y, radius)

        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
    else:
        target_state.clear()

    cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 1)
    cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 1)

    cv2.imshow("Drone Camera Feed", frame)
    cv2.waitKey(1)


def start_vision_thread():
    node = Node()
    success = node.subscribe(Image, CAMERA_TOPIC, camera_callback)
    if not success:
        print("Camera subscription failed — check topic name")
        return
    print(f"Subscribed to {CAMERA_TOPIC}")
    # gz-transport callback runs on its own thread internally; keep this thread alive
    while True:
        cv2.waitKey(100)


# ---------------- CONTROL: computes velocity from target_state ----------------
def compute_velocity_command(offset_x, offset_y, radius, detected):
    if not detected:
        return VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)

    # Yaw rate from horizontal offset
    if abs(offset_x) > FRAME_CENTER_DEADZONE:
        yaw_rate = np.clip(KP_YAW * offset_x, -MAX_YAW_RATE, MAX_YAW_RATE)
    else:
        yaw_rate = 0.0

    # Vertical speed from vertical offset (down_m_s positive = descend)
    if abs(offset_y) > FRAME_CENTER_DEADZONE:
        down_speed = np.clip(KP_VERTICAL * offset_y, -MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)
    else:
        down_speed = 0.0

    # Forward speed from radius error (bigger radius = closer = slow down/stop)
    radius_error = TARGET_RADIUS_PX - radius
    forward_speed = np.clip(KP_FORWARD * radius_error, 0.0, MAX_FORWARD_SPEED)
    if radius >= TARGET_RADIUS_PX:
        forward_speed = 0.0  # "arrived" - stop approaching

    return VelocityBodyYawspeed(forward_speed, 0.0, down_speed, yaw_rate)


# ---------------- MAIN: MAVSDK offboard loop ----------------
async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position OK")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.set_takeoff_altitude(5.0)
    await drone.action.takeoff()
    await asyncio.sleep(8)

    print("-- Setting initial setpoint")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard mode")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed: {error._result.result}")
        await drone.action.land()
        return

    print("-- Tracking loop running (Ctrl+C to stop)")
    last_detected_time = asyncio.get_event_loop().time()

    try:
        while True:
            offset_x, offset_y, radius, detected = target_state.get()
            now = asyncio.get_event_loop().time()

            if detected:
                last_detected_time = now
                cmd = compute_velocity_command(offset_x, offset_y, radius, True)
            else:
                # if we've been without detection too long, hold position
                if now - last_detected_time > LOST_TARGET_TIMEOUT:
                    cmd = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                else:
                    # brief dropout - hold last command briefly (here: just hover)
                    cmd = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)

            await drone.offboard.set_velocity_body(cmd)
            await asyncio.sleep(0.1)  # 10 Hz control loop

    except KeyboardInterrupt:
        pass
    finally:
        print("-- Stopping offboard mode")
        try:
            await drone.offboard.stop()
        except OffboardError as error:
            print(f"Stopping offboard mode failed: {error._result.result}")
        print("-- Landing")
        await drone.action.land()


def main():
    vision_thread = threading.Thread(target=start_vision_thread, daemon=True)
    vision_thread.start()

    asyncio.run(run())


if __name__ == "__main__":
    main()