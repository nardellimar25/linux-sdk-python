# udp_receivers.py

import socket
import threading
import cv2
import numpy as np
import time
import queue

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

class FrameReceiver(threading.Thread):
    """
    Receives JPEG frames via UDP, decodes them, saves a debug image,
    and puts the frame into the provided queue.
    """
    def __init__(self, ip, port, frame_queue, receiver_type,
                 debug_path, barrier, process_delay=0.01,
                 buffer_size=65535, debug=False):
        super().__init__(daemon=True)
        self.ip            = ip
        self.port          = port
        self.frame_queue   = frame_queue
        self.receiver_type = receiver_type   # "raw" or "blurred"
        self.debug_path    = debug_path
        self.barrier       = barrier
        self.process_delay = process_delay
        self.buffer_size   = buffer_size
        self.debug         = debug

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(1.0)

    def run(self):
        print(f"Listening for {self.receiver_type} frames on port {self.port}...")
        while True:
            try:
                data, _ = self.sock.recvfrom(self.buffer_size)
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    save_frame(self.debug_path, frame)
                    if self.debug:
                        print(f"[DEBUG] {self.receiver_type.capitalize()} JPEG → {self.debug_path}")
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                    self.barrier.wait(timeout=1)
                else:
                    print(f"Failed to decode frame in {self.receiver_type} receiver.")
            except (socket.timeout, threading.BrokenBarrierError):
                pass
            except Exception as e:
                print(f"Exception in {self.receiver_type} receiver: {e}")
            time.sleep(self.process_delay)

class CoordsReceiver(threading.Thread):
    """
    Receives coordinate data (in JSON format) via UDP and puts it into the provided queue.
    """
    def __init__(self, ip, port, coords_queue, barrier, process_delay=0.01, buffer_size=65535):
        super().__init__(daemon=True)
        self.ip            = ip
        self.port          = port
        self.coords_queue  = coords_queue
        self.barrier       = barrier
        self.process_delay = process_delay
        self.buffer_size   = buffer_size

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(1.0)

    def run(self):
        print(f"Listening for coordinate data on port {self.port}...")
        while True:
            try:
                data, _ = self.sock.recvfrom(self.buffer_size)
                coords_data = __import__('json').loads(data.decode('utf-8'))
                if self.coords_queue.full():
                    self.coords_queue.get_nowait()
                self.coords_queue.put(coords_data)
                self.barrier.wait(timeout=1)
            except (socket.timeout, threading.BrokenBarrierError):
                pass
            except Exception as e:
                print(f"Exception in coordinate receiver: {e}")
            time.sleep(self.process_delay)
