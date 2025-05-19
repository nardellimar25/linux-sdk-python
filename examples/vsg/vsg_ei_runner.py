# main.py
import threading
import queue
import time
import os
import cv2

from ultralytics import YOLO
from config_parser import Config
from classifier_worker import Classificator
from gst_receivers import VideoReceiver, MetaReceiver

class SimulatedReceiver(threading.Thread):
    """
    Simulates incoming frames by reading images from subfolders 'green', 'yellow', 'red'.
    Uses Ultralytics YOLO model (default weights) to detect persons and crops those regions for classification.
    """
    def __init__(self, raw_queue, coords_queue, config):
        super().__init__(daemon=True)
        self.raw_q    = raw_queue
        self.coords_q = coords_queue
        self.config   = config
        self.labels   = ['green', 'yellow', 'red']
        self.root     = config.TEST_IMAGES_PATH

        # Load default Ultralytics YOLO model (e.g., yolov8n)
        self.model = YOLO('yolov8n.pt')
        # Set detection parameters
        self.conf_threshold = 0.5
        self.iou_threshold  = 0.45

    def run(self):
        while True:
            for label in self.labels:
                folder = os.path.join(self.root, label)
                if not os.path.isdir(folder):
                    continue
                for fn in os.listdir(folder):
                    if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    img_path = os.path.join(folder, fn)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"[SIM] Failed to load image: {img_path}")
                        continue

                    # Perform YOLO inference
                    results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
                    for res in results:
                        for box in res.boxes:
                            # class 0 corresponds to 'person'
                            if int(box.cls) != 0:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            crop = img[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue

                            # Push the cropped frame
                            if self.raw_q.full():
                                self.raw_q.get_nowait()
                            self.raw_q.put(crop)

                            # Full-frame bbox of crop
                            h, w = crop.shape[:2]
                            self.coords_q.put({'bboxes': [(0, 0, w, h)]})

                            print(f"[SIM] Sent {label}/{fn} person crop to classifier")
                            time.sleep(self.config.PROCESS_DELAY)


def main():
    config = Config()

    raw_queue    = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    coords_queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)

    # Start the classification thread
    classifier = Classificator(raw_queue=raw_queue, coords_queue=coords_queue, config=config)
    classifier.start()

    if config.SIMULATED_INPUT:
        # Simulated input
        sim = SimulatedReceiver(raw_queue=raw_queue, coords_queue=coords_queue, config=config)
        sim.start()
    else:
        # Live input via GStreamer for all modes
        barrier = threading.Barrier(parties=2)

        vid_recv = VideoReceiver(
            config=config,
            raw_queue=raw_queue,
            barrier=barrier
        )
        vid_recv.start()

        meta_recv = MetaReceiver(
            config=config,
            coords_queue=coords_queue,
            barrier=barrier
        )
        meta_recv.start()

    print("Receiver running. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating...")


if __name__ == '__main__':
    main()