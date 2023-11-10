from typing import Optional

from ..adapters import *
from .base import Pipeline
from ..models.auto import MODEL_TO_ADAPTER
from ..utils.loading import StateDict

class TabularGenerationPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
        count: Optional[int] = None,
    ):
        state = StateDict.wrap(state)
        if count is None:
            count = 1 if state.train is None else len(state.train)
        if self.adapter is None:
            self.adapter = eval(MODEL_TO_ADAPTER[state.model.__class__])()
        state.synth = self.train_adapter.generate_data(
            count=count,
            model=state.model,
            **self.kwargs,
        )
        if self.output_format is not None:
            state.synth = self.ensure_output_format(state.synth)
        return state
