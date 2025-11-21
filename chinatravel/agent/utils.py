import sys
from numpy import ndarray, integer, floating
import numpy as np
import json


def decode_numpy_dict(d):
    if isinstance(d, dict):
        return {decode_numpy_dict(k): decode_numpy_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [decode_numpy_dict(i) for i in d]
    elif isinstance(d, integer):
        return int(d)
    elif isinstance(d, floating):
        return float(d)
    elif isinstance(d, ndarray):
        return decode_numpy_dict(d.tolist())
    else:
        return d


class Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout, debug_mode=False):

        self.debug_mode = debug_mode
        self.filename = filename
        self.log = open(filename, "a", encoding="utf-8")
        self.has_content = False

        if self.debug_mode:
            self.terminal = stream

    def write(self, message):
        if message.strip():  # Only write if message has content
            self.has_content = True
        self.log.write(message)

        if self.debug_mode:
            self.terminal.write(message)

    def flush(self):
        pass

    def __del__(self):
        self.log.close()
        # Remove empty log files
        import os
        if not self.has_content and os.path.exists(self.filename):
            try:
                if os.path.getsize(self.filename) == 0:
                    os.remove(self.filename)
            except:
                pass


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(json_data, file_path):
    # Create directory if it doesn't exist
    import os
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w", encoding="utf8") as dump_f:
        json.dump(json_data, dump_f, ensure_ascii=False, indent=4, cls=NpEncoder)
