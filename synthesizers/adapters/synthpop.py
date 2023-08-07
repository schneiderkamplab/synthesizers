from pandas import DataFrame
from synthpop import Synthpop

from .base import Adapter
from ..models.synthpop import SynthPopModel

MAP_TYPES = {
    'int64': 'int',
    'int8' : 'int',
    'category': 'category',
}

class SynthPopAdapter(Adapter):
    def __init__(self, input_formats=(DataFrame,)):
        super(SynthPopAdapter, self).__init__(input_formats)
    def train_model(self, data, plugin=None):
        data = self.ensure_input_format(data)
        dtypes = {col:MAP_TYPES[dtype.name] for col, dtype in data.dtypes.to_dict().items()}
        model = Synthpop()
        model.fit(data, dtypes)
        return SynthPopModel(model)
    def generate_data(self, count, model):
        result = model.model.generate(k=count)
        return result
    def evaluate_generated(self, original_data, generated_data):
        raise NotImplementedError("evaluation not implemented for SynthPop")
