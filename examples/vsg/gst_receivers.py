# gst_receivers.py

import gi, threading, queue, struct
import numpy as np
from udp_receivers import save_frame
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class Detection:
    """Parses a single detection struct from bytes."""
    _det_struct_pat = "B4H"
    byte_size = struct.Struct(_det_struct_pat).size

    def __init__(self, data):
        self.confidence, self.x1, self.y1, self.x2, self.y2 = struct.unpack(self._det_struct_pat, data)

class FrameMetadata:
    """Parses frame‐level metadata containing multiple Detection entries."""
    _count_struct = "H"
    det_count_byte_size = struct.Struct(_count_struct).size

    def __init__(self, data):
        # First two bytes = object count
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
    """
    Receives video frames via GStreamer, saves a debug image,
    and puts the frame into the provided queue.
    """
    def __init__(self, config, raw_queue, barrier):
        super().__init__(daemon=True)
        self.config    = config
        self.raw_queue = raw_queue
        self.barrier   = barrier

    def on_new_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf  = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w    = caps.get_value('width')
        h    = caps.get_value('height')

        success, info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.frombuffer(info.data, dtype=np.uint8).reshape((h, w, 3))
        buf.unmap(info)

        # save debug JPEG
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
            f'udpsrc address=0.0.0.0 port={self.config.UDP_PORT_RAW} caps="application/x-rtp, media=video, '
            'encoding-name=H264, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! '
            'video/x-raw, format=RGB ! appsink name=video_sink emit-signals=true max-buffers=1 drop=true'
        )
        pipeline = Gst.parse_launch(video_desc)
        sink     = pipeline.get_by_name("video_sink")
        sink.connect("new-sample", self.on_new_sample)

        pipeline.set_state(Gst.State.PLAYING)
        GLib.MainLoop().run()

class MetaReceiver(threading.Thread):
    """
    Receives metadata via GStreamer, parses bboxes, and puts them into the coords queue.
    """
    def __init__(self, config, coords_queue, barrier):
        super().__init__(daemon=True)
        self.config       = config
        self.coords_queue = coords_queue
        self.barrier      = barrier

    def on_new_meta_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf     = sample.get_buffer()
        success, info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        # COPY the bytes before unmapping
        data = bytes(info.data)
        buf.unmap(info)

        if len(data) >= FrameMetadata.det_count_byte_size:
            meta   = FrameMetadata(data)
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
            'caps="application/x-meta, media=meta" ! '
            'appsink name=meta_sink emit-signals=true max-buffers=1 drop=true'
        )
        pipeline = Gst.parse_launch(meta_desc)
        sink     = pipeline.get_by_name("meta_sink")
        sink.connect("new-sample", self.on_new_meta_sample)

        pipeline.set_state(Gst.State.PLAYING)
        GLib.MainLoop().run()
