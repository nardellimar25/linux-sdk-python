# main.py
import threading
import queue
import time

from config_parser import Config
from udp_receivers import FrameReceiver, CoordsReceiver
from classifier_worker import Classificator

def main():
    # Load configuration from config.ini
    config = Config()

    # Create queues with size defined in the config
    raw_queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    blurred_queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)
    coords_queue = queue.Queue(maxsize=config.QUEUE_MAX_SIZE)

    # Barrier to synchronize threads: raw, blurred, and coordinate receivers
    barrier = threading.Barrier(parties=3)

    # Create UDP receiver threads
    raw_receiver = FrameReceiver(
        ip=config.UDP_IP,
        port=config.UDP_PORT_RAW,
        frame_queue=raw_queue,
        receiver_type="raw",
        debug_path=config.RAW_DEBUG_PATH,
        barrier=barrier,
        process_delay=config.PROCESS_DELAY
    )

    blurred_receiver = FrameReceiver(
        ip=config.UDP_IP,
        port=config.UDP_PORT_BLURRED,
        frame_queue=blurred_queue,
        receiver_type="blurred",
        debug_path=config.BLUR_DEBUG_PATH,
        barrier=barrier,
        process_delay=config.PROCESS_DELAY
    )

    coords_receiver = CoordsReceiver(
        ip=config.UDP_IP,
        port=config.UDP_PORT_COORDS,
        coords_queue=coords_queue,
        barrier=barrier,
        process_delay=config.PROCESS_DELAY
    )

    # Create and start the classification thread
    classificator_thread = Classificator(
        raw_queue=raw_queue,
        blurred_queue=blurred_queue,
        coords_queue=coords_queue,
        config=config
    )

    # Start threads
    raw_receiver.start()
    blurred_receiver.start()
    coords_receiver.start()
    classificator_thread.start()

    print("Receiver running. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating...")

if __name__ == "__main__":
    main()
