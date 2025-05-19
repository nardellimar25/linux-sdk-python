# gst_receivers.py

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import threading
import struct
import time
import queue
import numpy as np
import cv2

def save_frame(file_path, frame, quality=80):
    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        print(f"[ERROR] Failed to encode frame for {file_path}.")
        return
    try:
        with open(file_path, 'wb') as f:
            f.write(jpeg.tobytes())
    except Exception as e:
        print(f"[ERROR] writing {file_path}: {e}")

class Detection:
    # uso big‐endian per corrispondere a struct.pack('>B4H', …)
    _det_struct_pat = ">B4H"
    byte_size      = struct.Struct(_det_struct_pat).size

    def __init__(self, data):
        self.confidence, self.x1, self.y1, self.x2, self.y2 = struct.unpack(self._det_struct_pat, data)

class FrameMetadata:
    _count_struct       = ">H"
    det_count_byte_size = struct.Struct(_count_struct).size

    def __init__(self, data):
        self.object_count = struct.unpack(self._count_struct, data[:self.det_count_byte_size])[0]
        self.objects = []
        offset = self.det_count_byte_size
        for _ in range(self.object_count):
            if offset + Detection.byte_size <= len(data):
                chunk = data[offset:offset + Detection.byte_size]
                self.objects.append(Detection(chunk))
                offset += Detection.byte_size
            else:
                break

class VideoReceiver(threading.Thread):
    def __init__(self, config, raw_queue, barrier):
        super().__init__(daemon=True)
        self.config    = config
        self.raw_queue = raw_queue
        self.barrier   = barrier

    def on_new_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        # ESTRAGGO LE CAPS DAL sample, non dal buffer
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_int('width')[1], caps.get_int('height')[1]

        buf = sample.get_buffer()
        success, info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        frame = np.frombuffer(info.data, dtype=np.uint8).reshape((h, w, 3))
        buf.unmap(info)

        save_frame(self.config.RAW_DEBUG_PATH, frame)
        if self.config.DEBUG:
            print(f"[DEBUG] Raw JPEG → {self.config.RAW_DEBUG_PATH}")

        if self.raw_queue.full():
            self.raw_queue.get_nowait()
        self.raw_queue.put(frame)

        try:
            self.barrier.wait(timeout=1)
        except:
            pass
        return Gst.FlowReturn.OK

    def run(self):
        Gst.init(None)
        video_desc = (
            f'udpsrc address=0.0.0.0 port={self.config.UDP_PORT_RAW} '
            'caps="application/x-rtp, media=video, encoding-name=H264, payload=96" '
            '! rtph264depay ! avdec_h264 ! videoconvert ! '
            'video/x-raw, format=RGB ! queue ! '
            'appsink name=video_sink emit-signals=true max-buffers=1 drop=true'
        )
        pipeline = Gst.parse_launch(video_desc)
        sink     = pipeline.get_by_name("video_sink")
        sink.connect("new-sample", self.on_new_sample)
        pipeline.set_state(Gst.State.PLAYING)
        GLib.MainLoop().run()

class MetaReceiver(threading.Thread):
    def __init__(self, config, coords_queue, barrier):
        super().__init__(daemon=True)
        self.config       = config
        self.coords_queue = coords_queue
        self.barrier      = barrier

    def on_new_meta_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        success, info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR
        data = bytes(info.data)
        buf.unmap(info)

        # Debug: stampa dimensione e count corretto
        print(f"[METADATA] Received {len(data)} bytes")
        meta = FrameMetadata(data)
        print(f"[METADATA] object_count = {meta.object_count}")

        bboxes = [(o.x1, o.y1, o.x2, o.y2) for o in meta.objects]
        if self.coords_queue.full():
            self.coords_queue.get_nowait()
        self.coords_queue.put({'bboxes': bboxes})

        try:
            self.barrier.wait(timeout=1)
        except:
            pass
        return Gst.FlowReturn.OK

    def run(self):
        Gst.init(None)
        meta_desc = (
            f'udpsrc address=0.0.0.0 port={self.config.UDP_PORT_COORDS} '
            'caps="application/x-meta, media=meta" ! queue ! '
            'appsink name=meta_sink emit-signals=true max-buffers=1 drop=true'
        )
        pipeline = Gst.parse_launch(meta_desc)
        sink     = pipeline.get_by_name("meta_sink")
        sink.connect("new-sample", self.on_new_meta_sample)
        pipeline.set_state(Gst.State.PLAYING)
        GLib.MainLoop().run()
