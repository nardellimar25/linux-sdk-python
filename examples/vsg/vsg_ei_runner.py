import threading
import queue
import time
import os
import cv2
import json

from ultralytics import YOLO
from config_parser import Config
from classifier_worker import Classificator
from gst_receivers import VideoReceiver, MetaReceiver

class FrameSaver(threading.Thread):
    """
    Thread that:
      1) drains display_queue (Edge Impulse outputs) and keeps only the latest frame,
      2) measures incoming FPS from Edge Impulse,
      3) dynamically adjusts its write delay so write-FPS ≃ 90% of Edge Impulse FPS,
      4) saves the frame for IIS to serve,
      5) writes a JSON metadata file with the current delay,
      6) logs both input and output FPS once per second.
    """
    def __init__(self, display_queue, stop_event, output_path, min_delay=0.01):
        super().__init__(daemon=True)
        self.display_queue = display_queue
        self.stop_event    = stop_event
        self.output_path   = output_path
        self.min_delay     = min_delay

        # Counters & timers for measuring FPS
        self.in_count   = 0
        self.in_start   = time.time()
        self.out_count  = 0
        self.out_start  = time.time()

        self.latest_frame  = None
        self.current_delay = min_delay

    def run(self):
        meta_path = os.path.splitext(self.output_path)[0] + '_meta.json'
        while not self.stop_event.is_set():
            now = time.time()

            # 1) Drain all frames from Edge Impulse; count them
            try:
                while True:
                    self.latest_frame = self.display_queue.get_nowait()
                    self.in_count += 1
            except queue.Empty:
                pass

            # 2) Every second, compute Edge Impulse FPS and update write delay
            if now - self.in_start >= 1.0:
                in_fps = self.in_count / (now - self.in_start)
                target_out_fps    = max(1.0, in_fps * 0.9)
                self.current_delay = 1.0 / target_out_fps
                print(f"[FrameSaver] EdgeImpulse-FPS: {in_fps:.1f}, write-delay: {self.current_delay*1000:.1f} ms")

                # Write metadata JSON for browser polling
                try:
                    with open(meta_path, 'w') as f:
                        json.dump({'delay_ms': round(self.current_delay * 1000, 1)}, f)
                except Exception as e:
                    print(f"[FrameSaver] ERROR writing meta file: {e}")

                self.in_count = 0
                self.in_start = now

            # 3) If we have a frame, write it to disk
            if self.latest_frame is not None:
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                success = cv2.imwrite(self.output_path, self.latest_frame)
                if not success:
                    print(f"[FrameSaver] ERROR writing frame to {self.output_path}")
                self.out_count += 1

            # 4) Every second, log actual write-FPS
            if now - self.out_start >= 1.0:
                out_fps = self.out_count / (now - self.out_start)
                print(f"[FrameSaver] write-FPS: {out_fps:.1f} fps")
                self.out_count = 0
                self.out_start = now

            # 5) Sleep the dynamically computed interval
            time.sleep(self.current_delay)

        print("FrameSaver stopping...")

class SimulatedReceiver(threading.Thread):
    """
    Simulates incoming frames by reading images from 'green', 'yellow', 'red' subfolders.
    Uses YOLO to detect persons, crops them, and sends them to the classifier.
    """
    def __init__(self, raw_queue, coords_queue, config, stop_event):
        super().__init__(daemon=True)
        self.raw_queue    = raw_queue
        self.coords_queue = coords_queue
        self.config       = config
        self.stop_event   = stop_event
        self.labels       = ['green', 'yellow', 'red']
        self.root         = config.TEST_IMAGES_PATH

        self.model          = YOLO('yolov8n.pt')
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

                for filename in os.listdir(folder):
                    if self.stop_event.is_set():
                        break
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"[SIM] Failed to load image: {img_path}")
                        continue

                    # Run YOLO detection on the image
                    results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
                    for res in results:
                        for box in res.boxes:
                            if int(box.cls) != 0:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            crop = img[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue

                            # Enqueue the crop and its bounding box
                            if self.raw_queue.full():
                                self.raw_queue.get_nowait()
                            self.raw_queue.put(crop)
                            self.coords_queue.put({
                                'bboxes': [(0, 0, crop.shape[1], crop.shape[0])]
                            })

                            print(f"[SIM] Sent {label}/{filename} person crop to classifier")
                            time.sleep(self.config.PROCESS_DELAY)
        print("SimulatedReceiver stopping...")

def main():
    config = Config()

    raw_queue     = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    coords_queue  = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    display_queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    stop_event    = threading.Event()

    # Start the classifier worker thread
    classifier = Classificator(
        raw_queue=raw_queue,
        coords_queue=coords_queue,
        display_queue=display_queue,
        config=config,
        stop_event=stop_event
    )
    classifier.start()

    # If demo mode is enabled, start the FrameSaver to write frames and metadata for IIS
    if config.DEMO:
        output_path = os.path.join(config.WWW_ROOT, config.FRAME_FILENAME)
        saver = FrameSaver(
            display_queue=display_queue,
            stop_event=stop_event,
            output_path=output_path,
            min_delay=config.FRAME_SAVE_DELAY
        )
        saver.start()

    # Start simulated input or real GStreamer receivers
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

    # Join all threads before exiting
    classifier.join()
    if config.DEMO:
        saver.join()
    if config.SIMULATED_INPUT:
        sim.join()
    else:
        vid_recv.join()
        meta_recv.join()

if __name__ == '__main__':
    main()
