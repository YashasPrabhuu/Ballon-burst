from picamera2 import Picamera2
import cv2
import numpy as np
import time
import re
import tensorflow as tf

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_PATH = "drone_detect.tflite"
LABELMAP_PATH = "label_map.pbtxt"
CONFIDENCE_THRESHOLD = 0.5


def load_label_map(path):
    label_map = {}
    with open(path, 'r') as f:
        content = f.read()
        items = re.findall(r'item\s*\{([^}]*)\}', content)
        for item in items:
            id_match = re.search(r'id\s*:\s*(\d+)', item)
            name_match = re.search(r'name\s*:\s*["\']([^"\']+)["\']', item)
            if id_match and name_match:
                label_map[int(id_match.group(1))] = name_match.group(1)
    return label_map


picam2 = Picamera2()
# Note: Picamera2's "RGB888" format actually returns BGR-ordered frames
# (the name refers to memory layout, not channel order). Requesting
# "BGR888" here means `frame` comes out already correct for OpenCV.
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
picam2.configure(config)
picam2.start()
time.sleep(2)

labels = load_label_map(LABELMAP_PATH)
print(f"Loaded labels: {labels}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_height = input_shape[1]
input_width = input_shape[2]
is_quantized = (input_details[0]['dtype'] == np.uint8)

print(f"Model input resolution: {input_width}x{input_height}")
for i, d in enumerate(output_details):
    print(f"Output {i}: shape={d['shape']}, dtype={d['dtype']}, quantization={d['quantization']}")

# Don't assume a fixed output order - different export pipelines put
# boxes/classes/scores/num_detections in different slots. Detect them
# by shape instead: boxes is the only tensor shaped (1, N, 4), and
# num_detections is the only one shaped (1,). The remaining two (1, N)
# tensors are classes and scores - distinguished below at runtime since
# shape alone can't tell them apart.
boxes_idx = None
num_idx = None
candidate_idx = []
for i, d in enumerate(output_details):
    shape = list(d['shape'])
    if len(shape) == 3 and shape[-1] == 4:
        boxes_idx = i
    elif len(shape) == 1:
        num_idx = i
    else:
        candidate_idx.append(i)

if boxes_idx is None or len(candidate_idx) != 2:
    raise RuntimeError(
        f"Could not identify output tensors automatically. "
        f"Got boxes_idx={boxes_idx}, num_idx={num_idx}, candidates={candidate_idx}. "
        f"Check the printed output shapes above and adjust the script manually."
    )

print(f"Detected: boxes=output{boxes_idx}, num_detections=output{num_idx}, "
      f"classes/scores candidates={candidate_idx}")

print("Starting TFLite inference - press 'q' to quit")


def dequantize(tensor, detail):
    """Convert raw quantized tensor values back to real-world scale.

    TFLite quantized models (dtype uint8/int8) store outputs as raw
    integers. These must be converted using: real_value = (raw - zero_point) * scale
    Without this step, box coordinates/scores will be near-meaningless
    small numbers, which shows up as a detection box stuck near (0,0).
    """
    tensor = np.asarray(tensor)
    if detail['dtype'] in (np.uint8, np.int8):
        scale, zero_point = detail['quantization']
        if scale == 0:
            return tensor.astype(np.float32)
        return (tensor.astype(np.float32) - zero_point) * scale
    return tensor.astype(np.float32)

fps_counter = 0
start_time = time.time()
fps = 0.0

try:
    while True:
        frame = picam2.capture_array()  # BGR, ready for display/OpenCV as-is
        image_height, image_width, _ = frame.shape

        # Most TFLite detection models expect RGB input, so convert just
        # for the model - the BGR `frame` itself is left untouched for display.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(rgb_frame, (input_width, input_height))

        if is_quantized:
            input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.uint8)
        else:
            input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
            input_tensor = (input_tensor - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        # Squeeze fully instead of just [0] indexing, since some exported
        # models produce extra singleton dimensions, which breaks plain
        # [0] indexing.
        boxes = np.squeeze(dequantize(interpreter.get_tensor(output_details[boxes_idx]['index']), output_details[boxes_idx]))

        cand_a = np.squeeze(dequantize(interpreter.get_tensor(output_details[candidate_idx[0]]['index']), output_details[candidate_idx[0]]))
        cand_b = np.squeeze(dequantize(interpreter.get_tensor(output_details[candidate_idx[1]]['index']), output_details[candidate_idx[1]]))
        cand_a = np.atleast_1d(cand_a)
        cand_b = np.atleast_1d(cand_b)

        # Classes are near-integer values (0.0, 1.0, 2.0...); scores are
        # continuous probabilities. Whichever candidate is "closer to
        # whole numbers" on average is treated as classes.
        a_integerness = np.mean(np.abs(cand_a - np.round(cand_a)))
        b_integerness = np.mean(np.abs(cand_b - np.round(cand_b)))
        if a_integerness < b_integerness:
            classes, scores = cand_a, cand_b
        else:
            classes, scores = cand_b, cand_a

        # A single detection can squeeze all the way down to 0-d arrays.
        # Normalize everything back to consistent shapes: boxes (N,4),
        # classes (N,), scores (N,).
        if scores.ndim == 0:
            scores = np.expand_dims(scores, axis=0)
            classes = np.expand_dims(classes, axis=0)
            boxes = np.expand_dims(boxes, axis=0)

        boxes = np.atleast_2d(boxes)
        classes = np.ravel(classes)
        scores = np.ravel(scores)

        display_frame = frame  # already BGR, no conversion needed
        h, w = display_frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        for i in range(len(scores)):
            score = float(scores[i])
            if score >= CONFIDENCE_THRESHOLD:
                box = np.ravel(boxes[i])
                ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

                left = int(xmin * image_width)
                top = int(ymin * image_height)
                right = int(xmax * image_width)
                bottom = int(ymax * image_height)

                class_id = int(np.ravel(classes[i])[0]) + 1
                class_name = labels.get(class_id, "drone")

                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                label = f"{class_name}: {int(score * 100)}%"
                cv2.putText(display_frame, label, (left, max(top - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.line(display_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 1)
        cv2.line(display_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 1)

        fps_counter += 1
        if time.time() - start_time >= 1.0:
            fps = fps_counter / (time.time() - start_time)
            fps_counter = 0
            start_time = time.time()

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Drone Detection (Picamera2)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
