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
    Thread that retrieves raw frames, blurred frames (or generates them in RENESAS mode),
    draws debug bounding boxes, classifies each region, and composites the final image.
    """
    def __init__(self, raw_queue, blurred_queue, coords_queue, config):
        super().__init__(daemon=True)
        self.raw_queue         = raw_queue
        self.blurred_queue     = blurred_queue
        self.coords_queue      = coords_queue
        self.config            = config
        self.active_image_path = config.ACTIVE_IMAGE_PATH
        self.coords_debug_path = config.COORDS_DEBUG_PATH
        self.process_delay     = config.PROCESS_DELAY

          # Initialize the Edge Impulse model
        if config.MODE == "NVIDIA":
            self.runner = ImageImpulseRunner(config.EDGE_IMPULSE_MODEL_PATH_NVIDIA)
        else:
            self.runner = ImageImpulseRunner(config.EDGE_IMPULSE_MODEL_PATH_RENESAS)
        self.model_info = self.runner.init()
        print(f"Model Info: {self.model_info}")

    def classify_image(self, image):
        """
        Convert the image to grayscale, resize to 96×96,
        flatten, and run the Edge Impulse classifier.
        """
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized  = cv2.resize(img_gray, (96, 96))
        return self.runner.classify(resized.flatten().tolist())

    def generate_blur_stream(self, raw_frame, bboxes):
        """
        For RENESAS mode when no blur UDP arrives,
        apply Gaussian blur to each bbox region on a copy of the raw frame.
        """
        blurred = raw_frame.copy()
        k = self.config.BLUR_KERNEL_SIZE
        for x1, y1, x2, y2 in bboxes:
            roi = raw_frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
            blurred[y1:y2, x1:x2] = blurred_roi
        return blurred

    def run(self):
        while True:
            try:
                coords_data = self.coords_queue.get(timeout=1)
                raw_frame   = get_latest_frame(self.raw_queue)
                if raw_frame is None:
                    continue

                # Obtain or generate the blurred frame
                if self.blurred_queue:
                    blurred_frame = get_latest_frame(self.blurred_queue)
                    if blurred_frame is None:
                        continue
                else:
                    blurred_frame = self.generate_blur_stream(
                        raw_frame, coords_data.get('bboxes', [])
                    )

                # Save and debug-print the blurred frame
                save_frame(self.config.BLUR_DEBUG_PATH, blurred_frame)
                if self.config.DEBUG:
                    print(f"[DEBUG] Blurred JPEG → {self.config.BLUR_DEBUG_PATH}")

                # Draw debug bounding boxes on raw frame
                debug_frame = raw_frame.copy()
                for x1, y1, x2, y2 in coords_data.get('bboxes', []):
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                save_frame(self.coords_debug_path, debug_frame)
                if self.config.DEBUG:
                    print(f"[DEBUG] BBoxes JPEG → {self.coords_debug_path}")

                # Create the final composite: replace 'green' regions with blurred regions
                final_frame = raw_frame.copy()
                for x1, y1, x2, y2 in coords_data.get('bboxes', []):
                    cropped = raw_frame[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue

                    start_time = time.time()
                    result     = self.classify_image(cropped)
                    inference_time = (time.time() - start_time) * 1000  # ms

                    cl_dict = result.get("result", {}).get("classification", {})
                    g = cl_dict.get("green", 0)
                    r = cl_dict.get("red",   0)
                    label = "green" if g >= r + 0.5 else "red"
                    confidence = cl_dict.get(label, 0)

                    print(f"Classification: {label.upper()} – Confidence: {confidence:.2f}, Time: {inference_time:.0f} ms")

                    if label == "green":
                        region = blurred_frame[y1:y2, x1:x2]
                        if region.shape == cropped.shape:
                            final_frame[y1:y2, x1:x2] = region

                # Save the composite result
                save_frame(self.active_image_path, final_frame)
                print(f"Saved composite frame to {self.active_image_path}")
                # debug-print the active frame path
                if self.config.DEBUG:
                    print(f"[DEBUG] Active JPEG → {self.active_image_path}")

            except (queue.Empty, threading.BrokenBarrierError):
                continue
            except Exception as e:
                print("Exception in classificator:", e)

            time.sleep(self.process_delay)
