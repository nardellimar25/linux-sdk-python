import os
import cv2
import time
import json
import threading
import queue
from edge_impulse_linux.image import ImageImpulseRunner

class Classificator(threading.Thread):
    def __init__(self, raw_queue, coords_queue, display_queue, config, stop_event, orient):
        super().__init__(daemon=True)
        self.raw_queue     = raw_queue
        self.coords_queue  = coords_queue
        self.display_queue = display_queue
        self.config        = config
        self.stop_event    = stop_event
        self.orient        = orient  # "left" o "right"

        # path per JPG e JSON
        self.WWW_ROOT     = config.WWW_ROOT
        self.frame_fname  = (config.FRAME_FILENAME_LEFT
                             if orient=='left'
                             else config.FRAME_FILENAME_RIGHT)
        self.json_fname   = self.frame_fname.replace('.jpg','_meta.json')
        self.frame_path   = os.path.join(self.WWW_ROOT, self.frame_fname)
        self.json_path    = os.path.join(self.WWW_ROOT, self.json_fname)

        # setup runner e blur
        model_path = (config.EDGE_IMPULSE_MODEL_PATH_NVIDIA
                      if config.MODE=='NVIDIA'
                      else config.EDGE_IMPULSE_MODEL_PATH_RENESAS)
        self.runner = ImageImpulseRunner(model_path)
        self.runner.init()
        self.blur_k = config.BLUR_KERNEL_SIZE

    def save_frame(self, path, frame, quality=80):
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ret:
            with open(path,'wb') as f: f.write(buf.tobytes())

    def write_meta(self, overall_label):
        try:
            with open(self.json_path,'w') as f:
                json.dump({'label': overall_label}, f)
        except Exception as e:
            print(f"[Classificator] ERROR writing JSON: {e}")

    def run(self):
        while not self.stop_event.is_set():
            try:
                coords = self.coords_queue.get(timeout=1)
            except queue.Empty:
                continue

            # prendi ultimo raw
            raw = None
            while not self.raw_queue.empty():
                raw = self.raw_queue.get_nowait()
            if raw is None:
                continue

            # blur debug
            blurred = raw.copy()
            for x1,y1,x2,y2 in coords.get('bboxes',[]):
                roi = raw[y1:y2, x1:x2]
                if roi.size:
                    blurred[y1:y2, x1:x2] = cv2.GaussianBlur(roi,(self.blur_k,self.blur_k),0)

            # classify & composizione finale
            final = raw.copy()
            labels = []
            for x1,y1,x2,y2 in coords.get('bboxes',[]):
                crop = raw[y1:y2, x1:x2]
                if crop.size==0: continue
                features,_ = self.runner.get_features_from_image_auto_studio_settings(
                               cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
                res = self.runner.classify(features)
                cls = res.get('result',{}).get('classification',{}) or {}
                label = max(('green','yellow','red'), key=lambda c: cls.get(c,0))
                labels.append(label)
                # disegna box + blur se green
                color = {'green':(0,255,0),'yellow':(0,255,255),'red':(0,0,255)}[label]
                cv2.rectangle(final,(x1,y1),(x2,y2),color,2)
                if label=='green':
                    final[y1:y2, x1:x2] = blurred[y1:y2, x1:x2]

            # decide overall
            if 'red' in labels:
                overall='red'
            elif 'yellow' in labels:
                overall='yellow'
            else:
                overall='green'

            # salva JPEG e JSON
            os.makedirs(self.WWW_ROOT,exist_ok=True)
            self.save_frame(self.frame_path, final)
            self.write_meta(overall)

            # push in display_queue
            try:
                self.display_queue.put_nowait(final)
            except queue.Full:
                pass

            time.sleep(self.config.PROCESS_DELAY)
