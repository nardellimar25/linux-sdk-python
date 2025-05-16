# classifier_worker.py

import cv2
import time
import threading
import queue
from edge_impulse_linux.image import ImageImpulseRunner

def save_frame(file_path, frame, quality=80):
    """
    Encode the frame to JPEG and save it to disk.
    """
    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ret:
        try:
            with open(file_path, 'wb') as f:
                f.write(jpeg.tobytes())
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
    else:
        print(f"Failed to encode frame for {file_path}.")

def get_latest_frame(q):
    """
    Extract and return the most recent frame from the queue, discarding older ones.
    """
    latest = None
    while not q.empty():
        try:
            latest = q.get_nowait()
        except queue.Empty:
            break
    return latest

class Classificator(threading.Thread):
    """
    Thread that pulls raw frames and bounding boxes, applies blur,
    preprocesses images exactly as Edge Impulse Studio does,
    runs classification, draws colored boxes, and saves the composite.
    """
    def __init__(self, raw_queue, coords_queue, config):
        super().__init__(daemon=True)
        self.raw_queue = raw_queue
        self.coords_queue = coords_queue
        self.config = config
        self.active_image_path = config.ACTIVE_IMAGE_PATH
        self.coords_debug_path = config.COORDS_DEBUG_PATH
        self.input_debug_path = getattr(config, 'INPUT_DEBUG_PATH', None)
        self.process_delay = config.PROCESS_DELAY
        self.blur_kernel_size = config.BLUR_KERNEL_SIZE

        # Initialize Edge Impulse runner
        model_path = (config.EDGE_IMPULSE_MODEL_PATH_NVIDIA 
                      if config.MODE == 'NVIDIA' 
                      else config.EDGE_IMPULSE_MODEL_PATH_RENESAS)
        self.runner = ImageImpulseRunner(model_path)
        self.model_info = self.runner.init()
        if config.DEBUG:
            print(f"[DEBUG] Initialized runner with model: {model_path}")

    def classify_image(self, image):
        """
        Preprocess the input image using the same pipeline as Edge Impulse Studio,
        extract features and classify directly.
        """
        # Convert from BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract features using auto-studio settings
        features, _ = self.runner.get_features_from_image_auto_studio_settings(img_rgb)

        # Classify by passing the raw feature array
        return self.runner.classify(features)

    def generate_blur(self, frame, bboxes):
        """
        Create a blurred copy of the frame, blurring only inside each bbox.
        """
        out = frame.copy()
        k = self.blur_kernel_size
        for x1, y1, x2, y2 in bboxes:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return out

    def run(self):
        while True:
            try:
                coords = self.coords_queue.get(timeout=1)
                raw = get_latest_frame(self.raw_queue)
                if raw is None:
                    continue

                # Debug: blurred image
                blurred = self.generate_blur(raw, coords.get('bboxes', []))
                save_frame(self.config.BLUR_DEBUG_PATH, blurred)
                if self.config.DEBUG:
                    print(f"[DEBUG] Blurred JPEG → {self.config.BLUR_DEBUG_PATH}")

                # Debug: draw original bboxes
                dbg = raw.copy()
                for x1, y1, x2, y2 in coords.get('bboxes', []):
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                save_frame(self.coords_debug_path, dbg)
                if self.config.DEBUG:
                    print(f"[DEBUG] BBoxes JPEG → {self.config.COORDS_DEBUG_PATH}")

                final = raw.copy()
                for x1, y1, x2, y2 in coords.get('bboxes', []):
                    crop = raw[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Debug: save classifier input
                    if self.input_debug_path:
                        save_frame(self.input_debug_path, crop)
                        if self.config.DEBUG:
                            print(f"[DEBUG] Classifier Input JPEG → {self.config.INPUT_DEBUG_PATH}")

                    # Classify
                    start = time.time()
                    try:
                        res = self.classify_image(crop)
                    except Exception as e:
                        print(f"[ERROR] classify_image failed: {e}")
                        continue
                    elapsed_ms = (time.time() - start) * 1000

                    # Parse results safely
                    cls = res.get('result', {}).get('classification', {}) or {}
                    g = cls.get('green', 0.0)
                    y_val = cls.get('yellow', 0.0)
                    r = cls.get('red', 0.0)
                    label = max(('green', 'yellow', 'red'), key=lambda c: cls.get(c, 0.0))
                    conf = cls.get(label, 0.0)

                    print(f"Classification → G:{g:.2f}, Y:{y_val:.2f}, R:{r:.2f} "
                          f"→ {label.upper()} ({conf:.2f}), {elapsed_ms:.0f}ms")

                    # Draw colored bbox and conditional blur
                    colors = {'green': (0, 255, 0), 'yellow': (0, 255, 255), 'red': (0, 0, 255)}
                    cv2.rectangle(final, (x1, y1), (x2, y2), colors[label], 2)
                    if label == 'green':
                        final[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]

                save_frame(self.active_image_path, final)
                print(f"Saved composite frame to {self.config.ACTIVE_IMAGE_PATH}")
                if self.config.DEBUG:
                    print(f"[DEBUG] Active JPEG → {self.config.ACTIVE_IMAGE_PATH}")

            except (queue.Empty, threading.BrokenBarrierError):
                continue
            except Exception as e:
                print(f"Exception in Classificator: {e}")

            time.sleep(self.process_delay)
