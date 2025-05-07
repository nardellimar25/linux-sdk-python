# main.py

import threading
import queue
import time

from config_parser import Config
from classifier_worker import Classificator

def main():
    config = Config()

    # Core queues
    raw_queue    = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    coords_queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)

    # Only in NVIDIA mode do we expect a blurred stream
    if config.MODE == "NVIDIA":
        from udp_receivers import FrameReceiver, CoordsReceiver
        blurred_queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
        barrier_parties = 3
    else:
        # RENESAS: use GStreamer receivers, no blurred_queue
        from gst_receivers import VideoReceiver as FrameReceiver, MetaReceiver as CoordsReceiver
        blurred_queue = None
        barrier_parties = 2

    barrier = threading.Barrier(parties=barrier_parties)

    # Instantiate frame receiver
    if config.MODE == "NVIDIA":
        raw_receiver = FrameReceiver(
            ip=config.UDP_IP,
            port=config.UDP_PORT_RAW,
            frame_queue=raw_queue,
            receiver_type="raw",
            debug_path=config.RAW_DEBUG_PATH,
            barrier=barrier,
            process_delay=config.PROCESS_DELAY,
            debug=config.DEBUG
        )
        raw_receiver.start()

        # Blurred receiver
        blurred_receiver = FrameReceiver(
            ip=config.UDP_IP,
            port=config.UDP_PORT_BLURRED,
            frame_queue=blurred_queue,
            receiver_type="blurred",
            debug_path=config.BLUR_DEBUG_PATH,
            barrier=barrier,
            process_delay=config.PROCESS_DELAY,
            debug=config.DEBUG
        )
        blurred_receiver.start()
    else:
        # RENESAS uses GStreamer VideoReceiver
        video_receiver = FrameReceiver(
            config=config,
            raw_queue=raw_queue,
            barrier=barrier
        )
        video_receiver.start()

    # Instantiate coords receiver (UDP or GStreamer)
    if config.MODE == "NVIDIA":
        coords_receiver = CoordsReceiver(
            ip=config.UDP_IP,
            port=config.UDP_PORT_COORDS,
            coords_queue=coords_queue,
            barrier=barrier,
            process_delay=config.PROCESS_DELAY
        )
    else:
        coords_receiver = CoordsReceiver(
            config=config,
            coords_queue=coords_queue,
            barrier=barrier
        )
    coords_receiver.start()

    # Start classification thread
    classificator = Classificator(
        raw_queue=raw_queue,
        blurred_queue=blurred_queue,
        coords_queue=coords_queue,
        config=config
    )
    classificator.start()

    print("Receiver running. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating...")

if __name__ == "__main__":
    main()
