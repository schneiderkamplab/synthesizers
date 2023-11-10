from ..utils import ensure_format

class Adapter:
    def __init__(self, input_formats):
        self.input_formats = input_formats
    def ensure_input_format(self, data, **kwargs):
        return ensure_format(data, self.input_formats, **kwargs)
