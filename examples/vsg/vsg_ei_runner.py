# main.py

import threading
import queue
import time

from config_parser import Config
from classifier_worker import Classificator

def main():
    config = Config()

    # Create queues
    raw_queue    = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    coords_queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)

    # Synchronization barrier
    parties = 2  # raw + coords only
    barrier  = threading.Barrier(parties=parties)

    if config.MODE == "NVIDIA":
        # Use UDP receivers
        from udp_receivers import FrameReceiver, CoordsReceiver

        raw_recv = FrameReceiver(
            ip=config.UDP_IP, port=config.UDP_PORT_RAW,
            frame_queue=raw_queue,
            debug_path=config.RAW_DEBUG_PATH,
            barrier=barrier,
            process_delay=config.PROCESS_DELAY,
            debug=config.DEBUG
        )
        raw_recv.start()

        coords_recv = CoordsReceiver(
            ip=config.UDP_IP, port=config.UDP_PORT_COORDS,
            coords_queue=coords_queue,
            barrier=barrier,
            process_delay=config.PROCESS_DELAY
        )
        coords_recv.start()

    else:
        # Use GStreamer receivers
        from gst_receivers import VideoReceiver, MetaReceiver

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

    # Start classification thread
    classifier = Classificator(
        raw_queue=raw_queue,
        coords_queue=coords_queue,
        config=config
    )
    classifier.start()

    print("Receiver running. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating...")

if __name__ == "__main__":
    main()
