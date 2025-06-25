import threading
import queue
import time
import os
import cv2
import json
from typing import Any

from ultralytics import YOLO
from config_parser import Config
from classifier_worker import Classificator
from gst_receivers import VideoReceiver, MetaReceiver


class SimulatedReceiver(threading.Thread):
    """
    Simulates an incoming video stream by iterating through images stored in
    the ``green``, ``yellow`` and ``red`` folders. For each image it runs
    YOLO‑v8 person detection, crops every detected person and enqueues both the
    crops and dummy bounding‑box metadata.
    """

    def __init__(
        self,
        raw_queue: queue.Queue[Any],
        coords_queue: queue.Queue[Any],
        config: Config,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.raw_queue = raw_queue
        self.coords_queue = coords_queue
        self.config = config
        self.stop_event = stop_event

        self.labels = ["green", "yellow", "red"]
        self.root = config.TEST_IMAGES_PATH

        self.model = YOLO("yolov8n.pt")
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45

    def run(self) -> None:
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
                    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
                    for res in results:
                        for box in res.boxes:
                            # Only keep the "person" class (ID 0)
                            if int(box.cls) != 0:
                                continue
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            crop = img[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue

                            # Drop the oldest element if the queue is full to keep latency bounded
                            if self.raw_queue.full():
                                try:
                                    self.raw_queue.get_nowait()
                                except queue.Empty:
                                    pass

                            self.raw_queue.put(crop)
                            # Dummy bbox covering the whole crop (some downstream stages expect it)
                            self.coords_queue.put({"bboxes": [(0, 0, crop.shape[1], crop.shape[0])]})

                            time.sleep(self.config.PROCESS_DELAY)
        print("SimulatedReceiver stopping…")


class FrameSaver(threading.Thread):
    """
    Writes the most recent frame from ``display_queue`` to *output_path* with
    minimal latency.

    * The thread blocks on ``queue.get`` (with a short timeout) so it wakes up
      immediately when a new frame is available – no artificial sleeps.
    * If *max_fps* is set, the saver will *skip* frames to respect that limit but
      will never insert delays.
    * Frames are encoded as JPEG with *jpeg_quality*.
    * If *atomic* is ``True`` (default) the frame is first written to a temporary
      file with the same extension (e.g. ``.tmp.jpg``) and then replaced
      atomically – readers never see a partially‑written file.
    """

    def __init__(
        self,
        display_queue: queue.Queue[Any],
        stop_event: threading.Event,
        output_path: str,
        jpeg_quality: int = 80,
        max_fps: float | None = None,
        atomic: bool = True,
        queue_timeout: float = 0.05,  # seconds
    ) -> None:
        super().__init__(daemon=True)

        self.q = display_queue
        self.stop_event = stop_event
        self.output_path = output_path
        # Use the same extension for the temporary file so OpenCV can pick a codec
        root, ext = os.path.splitext(output_path)
        self.tmp_path = f"{root}.tmp{ext}" if atomic else output_path
        self.atomic = atomic

        # OpenCV JPEG parameters – "optimize" disabled for speed
        self.jpeg_params = [
            cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,
        ]

        self.interval = (1.0 / max_fps) if max_fps else 0.0
        self.next_write = 0.0
        self.queue_timeout = queue_timeout

    def _write(self, frame: Any) -> None:
        """Encode and write a single frame to disk."""
        # Write to tmp‑file first (correct extension ensures a valid codec)
        cv2.imwrite(self.tmp_path, frame, self.jpeg_params)
        if self.atomic and self.tmp_path != self.output_path:
            # ``os.replace`` is atomic on POSIX; viewers never read half a frame.
            os.replace(self.tmp_path, self.output_path)

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.q.get(timeout=self.queue_timeout)
            except queue.Empty:
                continue

            now = time.time()
            if self.interval and now < self.next_write:
                # Skip this frame to honour max_fps
                continue

            self._write(frame)
            self.next_write = now + self.interval
        print("FrameSaver stopping…")


def main() -> None:
    config = Config()

    model_path = (
        config.EDGE_IMPULSE_MODEL_PATH_NVIDIA
        if config.MODE == "NVIDIA"
        else config.EDGE_IMPULSE_MODEL_PATH_RENESAS
    )
    orient = "left" if "left" in model_path.lower() else "right"

    raw_q: queue.Queue[Any] = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    coords_q: queue.Queue[Any] = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    display_q: queue.Queue[Any] = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    stop_evt = threading.Event()

    # Classifier worker
    classifier = Classificator(
        raw_queue=raw_q,
        coords_queue=coords_q,
        display_queue=display_q,
        config=config,
        stop_event=stop_evt,
        orient=orient,
    )
    classifier.start()

    # FrameSaver (demo mode only)
    if config.DEMO:
        fname = (
            config.FRAME_FILENAME_LEFT if orient == "left" else config.FRAME_FILENAME_RIGHT
        )
        path = os.path.join(config.WWW_ROOT, fname)

        saver = FrameSaver(
            display_queue=display_q,
            stop_event=stop_evt,
            output_path=path,
            jpeg_quality=getattr(config, "JPEG_QUALITY", 80),
            max_fps=getattr(config, "FRAME_SAVE_MAX_FPS", None),  # None → save every frame
            atomic=True,
        )
        saver.start()

    # Input receivers
    if config.SIMULATED_INPUT:
        sim = SimulatedReceiver(raw_q, coords_q, config, stop_evt)
        sim.start()
    else:
        barrier = threading.Barrier(2)
        vid = VideoReceiver(config, raw_q, barrier, stop_evt)
        vid.start()
        meta = MetaReceiver(config, coords_q, barrier, stop_evt)
        meta.start()

    print("Receiver running. Press Ctrl+C to terminate.")
    try:
        while not stop_evt.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_evt.set()

    # Graceful shutdown
    classifier.join()
    if config.DEMO:
        saver.join()
    if config.SIMULATED_INPUT:
        sim.join()
    else:
        vid.join()
        meta.join()


if __name__ == "__main__":
    main()
