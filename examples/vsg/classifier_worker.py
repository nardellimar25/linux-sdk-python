# classifier_worker.py

import cv2
import time
import threading
import queue
from edge_impulse_linux.image import ImageImpulseRunner

def save_frame(file_path, frame, quality=80):
    """Encode the frame to JPEG and save it to disk."""
    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ret:
        try:
            with open(file_path, "wb") as f:
                f.write(jpeg.tobytes())
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
    else:
        print(f"Failed to encode frame for {file_path}.")

def get_latest_frame(q):
    """Extract and return the last available frame from the queue."""
    latest = None
    while not q.empty():
        try:
            latest = q.get_nowait()
        except queue.Empty:
            break
    return latest

class Classificator(threading.Thread):
    """
    Thread that retrieves raw frames and coords, generates a blurred version in-process,
    runs classification on each bbox, and composites the final image.
    """
    def __init__(self, raw_queue, coords_queue, config):
        super().__init__(daemon=True)
        self.raw_queue         = raw_queue
        self.coords_queue      = coords_queue
        self.config            = config
        self.active_image_path = config.ACTIVE_IMAGE_PATH
        self.coords_debug_path = config.COORDS_DEBUG_PATH
        self.process_delay     = config.PROCESS_DELAY
        self.blur_kernel_size = config.BLUR_KERNEL_SIZE

          # Initialize the Edge Impulse model
        if config.MODE == "NVIDIA":
            self.runner = ImageImpulseRunner(config.EDGE_IMPULSE_MODEL_PATH_NVIDIA)
        else:
            self.runner = ImageImpulseRunner(config.EDGE_IMPULSE_MODEL_PATH_RENESAS)
        self.model_info = self.runner.init()
        print(f"Model Info: {self.model_info}")


    def classify_image(self, image):
        """Preprocess and classify with Edge Impulse."""
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (96, 96))
        return self.runner.classify(resized.flatten().tolist())

    def generate_blur(self, raw, bboxes):
        """Apply Gaussian blur over each bbox on a copy of the raw frame."""
        out = raw.copy()
        k   = self.blur_kernel_size
        for x1, y1, x2, y2 in bboxes:
            roi = raw[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return out

    def run(self):
        while True:
            try:
                coords = self.coords_queue.get(timeout=1)
                raw    = get_latest_frame(self.raw_queue)
                if raw is None:
                    continue

                # Generate blur in-process for both modes
                blurred = self.generate_blur(raw, coords.get('bboxes', []))
                save_frame(self.config.BLUR_DEBUG_PATH, blurred)
                if self.config.DEBUG:
                    print(f"[DEBUG] Blurred JPEG → {self.config.BLUR_DEBUG_PATH}")

                # Draw debug bboxes
                dbg = raw.copy()
                for x1, y1, x2, y2 in coords.get('bboxes', []):
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0,255,0), 2)
                save_frame(self.coords_debug_path, dbg)
                if self.config.DEBUG:
                    print(f"[DEBUG] BBoxes JPEG → {self.config.COORDS_DEBUG_PATH}")

                # Composite based on classification
                final = raw.copy()
                for x1, y1, x2, y2 in coords.get('bboxes', []):
                    crop = raw[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    t0     = time.time()
                    res    = self.classify_image(crop)
                    t_ms   = (time.time() - t0) * 1000
                    cls    = res.get("result", {}).get("classification", {})
                    g, r   = cls.get("green",0), cls.get("red",0)
                    label  = "green" if g >= r+0.5 else "red"
                    conf   = cls.get(label,0)

                    print(f"Classification: {label.upper()} – Confidence: {conf:.2f}, Time: {t_ms:.0f} ms")

                    if label=="green":
                        roi = blurred[y1:y2, x1:x2]
                        if roi.shape==crop.shape:
                            final[y1:y2, x1:x2] = roi

                save_frame(self.active_image_path, final)
                print(f"Saved composite frame to {self.config.ACTIVE_IMAGE_PATH}")
                if self.config.DEBUG:
                    print(f"[DEBUG] Active JPEG → {self.config.ACTIVE_IMAGE_PATH}")

            except (queue.Empty, threading.BrokenBarrierError):
                continue
            except Exception as e:
                print("Exception in classificator:", e)

            time.sleep(self.process_delay)
