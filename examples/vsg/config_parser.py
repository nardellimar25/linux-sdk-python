# config_parser.py

import configparser

class Config:
    def __init__(self, file_path="config.ini"):
        self.parser = configparser.ConfigParser()
        self.parser.read(file_path)
        self._parse_config()

    def _parse_config(self):
        # Network Section
        self.UDP_IP           = self.parser.get("Network", "UDP_IP")
        self.UDP_PORT_RAW     = self.parser.getint("Network", "UDP_PORT_RAW")
        self.UDP_PORT_BLURRED = self.parser.getint("Network", "UDP_PORT_BLURRED")
        self.UDP_PORT_COORDS  = self.parser.getint("Network", "UDP_PORT_COORDS")

        # Paths Section
        self.RAW_DEBUG_PATH    = self.parser.get("Paths", "RAW_DEBUG_PATH")
        self.BLUR_DEBUG_PATH   = self.parser.get("Paths", "BLUR_DEBUG_PATH")
        self.COORDS_DEBUG_PATH = self.parser.get("Paths", "COORDS_DEBUG_PATH")
        self.ACTIVE_IMAGE_PATH = self.parser.get("Paths", "ACTIVE_IMAGE_PATH")

        # Model Section
        self.EDGE_IMPULSE_MODEL_PATH_NVIDIA = self.parser.get("Model", "EDGE_IMPULSE_MODEL_PATH_NVIDIA")
        self.EDGE_IMPULSE_MODEL_PATH_RENESAS = self.parser.get("Model", "EDGE_IMPULSE_MODEL_PATH_RENESAS")

        # General Section
        self.PROCESS_DELAY  = self.parser.getfloat("General", "PROCESS_DELAY")
        self.QUEUE_MAX_SIZE = self.parser.getint("General", "QUEUE_MAX_SIZE")
        self.DEBUG          = self.parser.getboolean("General", "DEBUG")

        # Device Section
        self.MODE = self.parser.get("Device", "MODE").upper()
        if self.MODE == "RENESAS":
            self.BLUR_KERNEL_SIZE = self.parser.getint("Renesas", "BLUR_KERNEL_SIZE")
