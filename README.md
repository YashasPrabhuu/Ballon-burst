# Ballon-burst

Vision-guided drone that finds a red balloon and flies toward it. The project has two halves:

- **Simulation** – PX4 SITL + Gazebo (`x500_mono_cam` quadcopter in the `baylands` world). OpenCV detects the balloon from the simulated camera, and MAVSDK steers the drone toward it.
- **Real hardware** – Raspberry Pi with a Pi Camera (`picamera2`) running either colour-based balloon detection or a TFLite object detector, plus MAVSDK over serial to a flight controller.

All code lives in [`Sample/`](Sample/).

## Contents of `Sample/`

| File | Runs on | What it does |
|---|---|---|
| `sample.py` | Sim | Smoke test: connect, arm, take off to 5 m, hover, land. |
| `cameratopic.py` | Sim | Shows the Gazebo drone camera feed in an OpenCV window. |
| `ballontrack.py` | Sim | Detects a red balloon in the Gazebo camera feed and prints its position/offset from frame centre. No flight control. |
| `towardsballon.py` | Sim | Full pipeline: takes off, then yaws/climbs/approaches the balloon using offboard velocity commands. **Flies the drone.** |
| `offboard_velocity.py` | Real HW | Takeoff, move forward, yaw, land via offboard mode over `/dev/serial0` @ 921600. **Flies the drone.** |
| `camerpi.py` | Pi | Red balloon detection from the Pi Camera, with a debug HSV mask window. |
| `camera_infer.py` | Pi | TFLite object detection (`drone_detect.tflite`) on the Pi Camera feed with FPS overlay. |
| `drone_detect.tflite`, `drone_detect_int8.tflite` | Pi | Float and int8-quantised detection models. |
| `label_map.pbtxt` | Pi | Class labels (`drones`, id 1) used by `camera_infer.py`. |

## Prerequisites

### Common
- Python 3.10+
- `pip install numpy opencv-python mavsdk`

### Simulation (Ubuntu 22.04/24.04 recommended)
- [PX4-Autopilot](https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html) built from source
- Gazebo (gz) with Python bindings providing `gz.transport13` and `gz.msgs10`
  (on Ubuntu: `sudo apt install python3-gz-transport13 python3-gz-msgs10`; if you use a virtualenv, create it with `--system-site-packages` so these are visible)
- A GUI session (the scripts open OpenCV windows)

### Raspberry Pi
- Raspberry Pi OS with a Pi Camera enabled
- `sudo apt install python3-picamera2` (use a venv with `--system-site-packages` so `picamera2` is visible)
- `pip install numpy opencv-python tflite-runtime` (or `tensorflow`; `camera_infer.py` imports `tensorflow as tf`)
- For flight: flight controller wired to the Pi's serial port (`/dev/serial0`), with serial enabled and the console disabled via `raspi-config`

## Quick start – simulation

1. **Start PX4 SITL with the camera drone** (from your PX4-Autopilot directory):
   ```bash
   PX4_GZ_WORLD=baylands make px4_sitl gz_x500_mono_cam
   ```
   Wait until Gazebo opens and PX4 reports it is ready.

2. **Add a red balloon** (any red sphere model) to the world in Gazebo so the camera has something to see.

3. **Verify the camera feed** in a second terminal:
   ```bash
   cd Sample
   python3 cameratopic.py
   ```
   If you get "Subscription failed", list the topics with `gz topic -l` and update `TOPIC` / `CAMERA_TOPIC` in the scripts. They default to
   `/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image`.

4. **Check balloon detection** (no flight):
   ```bash
   python3 ballontrack.py
   ```

5. **Test basic flight** (optional):
   ```bash
   python3 sample.py
   ```

6. **Run the full balloon-chasing pipeline**:
   ```bash
   python3 towardsballon.py
   ```
   The drone arms, takes off to 5 m, enters offboard mode and steers toward the balloon. It stops approaching once the balloon's apparent radius reaches `TARGET_RADIUS_PX` (80 px). Press `Ctrl+C` to stop offboard mode and land.

## Quick start – Raspberry Pi

Run from `Sample/` (the scripts load the model and label map by relative path).

```bash
cd Sample

# Colour-based red balloon detection (press q to quit)
python3 camerpi.py

# TFLite detector with FPS overlay (press q to quit)
python3 camera_infer.py
```

To use the quantised model, change `MODEL_PATH` in `camera_infer.py` to `drone_detect_int8.tflite`.

To fly from the Pi, connect the flight controller and run:

```bash
python3 offboard_velocity.py
```

> **Safety:** this arms the vehicle and flies it. Remove propellers for bench tests, fly in an open area with a safety pilot ready to take over, and test in simulation first.

## Configuration

Tunable values sit at the top of each script:

- **Connection address** – `udp://:14540` for SITL (`sample.py`, `towardsballon.py`); `serial:///dev/serial0:921600` for hardware (`offboard_velocity.py`).
- **Detection** – the red HSV ranges in `detect_balloon()` (`[0,120,70]–[10,255,255]` and `[170,120,70]–[180,255,255]`) and `MIN_RADIUS_TO_TRACK`. Use the mask window in `camerpi.py` to tune them for your lighting.
- **Control** (`towardsballon.py`) – `KP_YAW`, `KP_VERTICAL`, `KP_FORWARD`, speed limits, `FRAME_CENTER_DEADZONE`, `TARGET_RADIUS_PX`.
- **Model** (`camera_infer.py`) – `MODEL_PATH`, `CONFIDENCE_THRESHOLD`, frame size.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: gz.transport13` | Install the Gazebo Python bindings and make them visible to your Python environment (see prerequisites). |
| `Subscription failed` | Wrong topic name or Gazebo not running. Check `gz topic -l`. |
| Script hangs on "Waiting for drone to connect..." | PX4 SITL isn't running, or the connection address doesn't match. |
| Offboard mode fails to start | An initial setpoint must be sent first (already done in the scripts); check the drone is airborne and has a position estimate. |
| `picamera2` not found | Install via apt and use `--system-site-packages` in your venv. |
| Colours look swapped on the Pi | Picamera2's `RGB888` format already yields BGR-ordered frames; `camerpi.py` additionally applies `cv2.COLOR_RGB2BGR`, so remove that conversion if red/blue are inverted. |
| `camera_infer.py` can't find the model | Run it from inside `Sample/`. |

## License

See [LICENSE](LICENSE).
