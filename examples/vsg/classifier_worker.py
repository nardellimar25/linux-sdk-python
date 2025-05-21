# classifier_worker.py
import cv2
import time
import threading
import queue
from edge_impulse_linux.image import ImageImpulseRunner


def save_frame(file_path, frame, quality=80):
    # Encode and save JPEG
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
    # Drain queue and return the most recent frame
    latest = None
    while not q.empty():
        try:
            latest = q.get_nowait()
        except queue.Empty:
            break
    return latest


class Classificator(threading.Thread):
    """
    Worker thread that classifies person crops, applies blur based on label,
    draws bounding boxes, saves to disk, and enqueues the final image for display.
    """
    def __init__(self, raw_queue, coords_queue, display_queue, config, stop_event):
        super().__init__(daemon=True)
        self.raw_queue     = raw_queue
        self.coords_queue  = coords_queue
        self.display_queue = display_queue
        self.config        = config
        self.active_image_path = config.ACTIVE_IMAGE_PATH
        self.coords_debug_path = config.COORDS_DEBUG_PATH
        self.input_debug_path = config.INPUT_DEBUG_PATH
        self.stop_event = stop_event
        
        self.blur_kernel_size = config.BLUR_KERNEL_SIZE
        # Initialize Edge Impulse model runner
        model_path = (config.EDGE_IMPULSE_MODEL_PATH_NVIDIA 
                      if config.MODE == 'NVIDIA' 
                      else config.EDGE_IMPULSE_MODEL_PATH_RENESAS)
        self.runner = ImageImpulseRunner(model_path)
        self.runner.init()
        if config.DEBUG:
            print(f"[DEBUG] Initialized runner with model: {model_path}")


    def classify_image(self, image):
        # Convert BGR to RGB and extract features/classify
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        features, _ = self.runner.get_features_from_image_auto_studio_settings(img_rgb)
        return self.runner.classify(features)

    def generate_blur(self, frame, bboxes):
        out = frame.copy()
        k = self.blur_kernel_size
        for x1, y1, x2, y2 in bboxes:
            roi = frame[y1:y2, x1:x2]
            if roi.size:
                out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return out

    def run(self):
        while not self.stop_event.is_set():
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

                    try:
                        # Classify crop and parse results
                        res = self.classify_image(crop)
                        cls = res.get('result', {}).get('classification', {}) or {}
                        label = max(('green','yellow','red'), key=lambda c: cls.get(c, 0.0))
                    except Exception as e:
                        print(f"[ERROR] Classification failed: {e}")
                        continue

                    # Draw colored box
                    colors = {'green': (0,255,0), 'yellow': (0,255,255), 'red': (0,0,255)}
                    cv2.rectangle(final, (x1,y1), (x2,y2), colors[label], 2)
                    # If green, overlay blur
                    if label == 'green':
                        final[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]

                # Save composite to disk
                save_frame(self.config.ACTIVE_IMAGE_PATH, final)

                # Enqueue for display without blocking
                try:
                    self.display_queue.put_nowait(final)
                except queue.Full:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Exception in Classificator: {e}")
            time.sleep(self.config.PROCESS_DELAY)
