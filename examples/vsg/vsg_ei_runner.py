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

class DemoDisplay(threading.Thread):
    """
    Thread that displays the processed video stream in a single window.
    It reads fully processed frames from `display_queue` and shows them.
    Uses cv2.startWindowThread to avoid QBasicTimer warnings when using Qt backend.
    """
    def __init__(self, display_queue, stop_event):
        super().__init__(daemon=True)
        self.disp_q = display_queue
        self.stop_event = stop_event

    def run(self):
        cv2.startWindowThread()
        window_name = "VSG Video"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while not self.stop_event.is_set():
            try:
                frame = self.disp_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                continue

            cv2.imshow(window_name, frame)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


class SimulatedReceiver(threading.Thread):
    """
    Simulates incoming frames by reading images from subfolders 'green', 'yellow', 'red'.
    Uses Ultralytics YOLO model (default weights) to detect persons and crops those regions for classification.
    """
    def __init__(self, raw_queue, coords_queue, config, stop_event):
        super().__init__(daemon=True)
        self.raw_q    = raw_queue
        self.coords_q = coords_queue
        self.config   = config
        self.stop_event = stop_event
        self.labels   = ['green', 'yellow', 'red']
        self.root     = config.TEST_IMAGES_PATH

        self.model = YOLO('yolov8n.pt')
        self.conf_threshold = 0.5
        self.iou_threshold  = 0.45

    def run(self):
        while not self.stop_event.is_set():
            for label in self.labels:
                if self.stop_event.is_set():
                    break
                folder = os.path.join(self.root, label)
                if not os.path.isdir(folder):
                    continue
                for fn in os.listdir(folder):
                    if self.stop_event.is_set():
                        break
                    if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    img_path = os.path.join(folder, fn)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"[SIM] Failed to load image: {img_path}")
                        continue

                    results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
                    for res in results:
                        for box in res.boxes:
                            if int(box.cls) != 0:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            crop = img[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue

                            if self.raw_q.full():
                                self.raw_q.get_nowait()
                            self.raw_q.put(crop)
                            self.coords_q.put({'bboxes': [(0, 0, crop.shape[1], crop.shape[0])]})

                            print(f"[SIM] Sent {label}/{fn} person crop to classifier")
                            time.sleep(self.config.PROCESS_DELAY)
        print("SimulatedReceiver stopping...")


def main():
    config = Config()

    raw_queue     = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    coords_queue  = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    display_queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)

    stop_event = threading.Event()

    # Start the classification thread
    classifier = Classificator(
        raw_queue=raw_queue,
        coords_queue=coords_queue,
        display_queue=display_queue,
        config=config,
        stop_event=stop_event
    )
    classifier.start()

    if config.DEMO:
        demo = DemoDisplay(display_queue, stop_event)
        demo.start()

    if config.SIMULATED_INPUT:
        sim = SimulatedReceiver(
            raw_queue=raw_queue,
            coords_queue=coords_queue,
            config=config,
            stop_event=stop_event
        )
        sim.start()
    else:
        barrier = threading.Barrier(parties=2)

        vid_recv = VideoReceiver(
            config=config,
            raw_queue=raw_queue,
            barrier=barrier,
            stop_event=stop_event
        )
        vid_recv.start()

        meta_recv = MetaReceiver(
            config=config,
            coords_queue=coords_queue,
            barrier=barrier,
            stop_event=stop_event
        )
        meta_recv.start()

    print("Receiver running. Press Ctrl+C to terminate.")
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating...")
        stop_event.set()

    # Join threads
    classifier.join()
    if config.DEMO:
        demo.join()
    if config.SIMULATED_INPUT:
        sim.join()
    else:
        vid_recv.join()
        meta_recv.join()

if __name__ == '__main__':
    main()
