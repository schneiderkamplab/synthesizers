from typing import Optional

from ..adapters import *
from .base import Pipeline
from ..models.auto import MODEL_TO_ADAPTER
from ..utils.formats import StateDict

class TabularGenerationPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
        count: Optional[int] = None,
    ):
        state = StateDict.wrap(state)
        kwargs = dict(self.kwargs)
        kwargs.update(self.gen_args)
        if count is None:
            if kwargs.get("count", None) is None:
                kwargs["count"] = 1 if state.train is None else len(state.train)
        else:
            kwargs["count"] = count
        if self.train_adapter is None:
            self.train_adapter = eval(MODEL_TO_ADAPTER[state.model.__class__])()
        state.synth = self.train_adapter.generate_data(
            model=state.model,
            **kwargs,
        )
        if self.output_format is not None:
            state.synth = self.ensure_output_format(state.synth)
        if self.save_args.get("name", None) is not None:
            state.Save(**self.save_args)
        return state
