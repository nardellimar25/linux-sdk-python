# classificator.py
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
            with open(file_path, "wb") as f:
                f.write(jpeg.tobytes())
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
    else:
        print(f"Failed to encode frame for {file_path}.")

def get_latest_frame(q):
    """
    Extract and return the last available frame from the queue.
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
    Thread that retrieves raw frames, blurred frames, and coordinates from the queues,
    draws debug bounding boxes, and for each bounding box classifies the corresponding region.
    If the classification returns "green", it replaces that region in the final composite image.
    """
    def __init__(self, raw_queue, blurred_queue, coords_queue, config):
        super().__init__(daemon=True)
        self.raw_queue = raw_queue
        self.blurred_queue = blurred_queue
        self.coords_queue = coords_queue
        self.active_image_path = config.ACTIVE_IMAGE_PATH
        self.coords_debug_path = config.COORDS_DEBUG_PATH
        self.process_delay = config.PROCESS_DELAY

        # Initialize the Edge Impulse model
        self.runner = ImageImpulseRunner(config.EDGE_IMPULSE_MODEL_PATH)
        self.model_info = self.runner.init()
        print(f"Model Info: {self.model_info}")

    def classify_image(self, image):
        """
        Convert image to grayscale, resize to 96x96, flatten, and classify using Edge Impulse.
        """
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img_gray, (96, 96))
        result = self.runner.classify(resized.flatten().tolist())
        return result

    def run(self):
        while True:
            try:
                coords_data = self.coords_queue.get(timeout=1)
                raw_frame = get_latest_frame(self.raw_queue)
                blurred_frame = get_latest_frame(self.blurred_queue)

                if raw_frame is None or blurred_frame is None:
                    continue

                # Draw bounding boxes on the raw frame for debugging
                bbox_frame = raw_frame.copy()
                for bbox in coords_data.get('bboxes', []):
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                save_frame(self.coords_debug_path, bbox_frame)

                # Create the composite image starting from the raw frame
                final_frame = raw_frame.copy()
                for bbox in coords_data.get('bboxes', []):
                    x1, y1, x2, y2 = bbox
                    cropped_img = raw_frame[y1:y2, x1:x2]
                    if cropped_img.size == 0:
                        continue

                    start_time = time.time()
                    classification = self.classify_image(cropped_img)
                    inference_time = (time.time() - start_time) * 1000  # in ms

                    cl_dict = classification.get("result", {}).get("classification", {})
                    conf_green = cl_dict.get("green", 0)
                    conf_red = cl_dict.get("red", 0)
                    label = "green" if conf_green >= conf_red + 0.5 else "red"
                    confidence = cl_dict.get(label, 0)

                    print(f"Classification: {label.upper()} – Confidence: {confidence:.2f}, Time: {inference_time:.0f} ms")

                    if label == "green":
                        blurred_region = blurred_frame[y1:y2, x1:x2]
                        if blurred_region.shape == cropped_img.shape:
                            final_frame[y1:y2, x1:x2] = blurred_region

                save_frame(self.active_image_path, final_frame)
                print(f"Saved composite frame to {self.active_image_path}")

            except (queue.Empty, threading.BrokenBarrierError):
                continue
            except Exception as e:
                print("Exception in classificator:", e)
            time.sleep(self.process_delay)
