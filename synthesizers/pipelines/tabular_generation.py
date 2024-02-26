from itertools import chain
from typing import List, Optional, Union

from ..adapters import *
from .base import Pipeline
from ..models.auto import MODEL_TO_ADAPTER
from ..utils.formats import State, StateDict

class TabularGenerationPipeline(Pipeline):

    def _call(
        self,
        state: StateDict,
        count: Optional[Union[int, List[int]]] = None,
    ) -> List[StateDict]:
        kwargs = dict(self.kwargs)
        kwargs.update(self.gen_args)
        if count is None:
            if kwargs.get("count", None) is None:
                kwargs["count"] = 1 if state.train is None else len(state.train)
        else:
            kwargs["count"] = count
        if self.train_adapter is None:
            self.train_adapter = eval(MODEL_TO_ADAPTER[state.model.__class__])()
        if isinstance(kwargs["count"], list):
            state_dicts = (self._call(state.clone(), count=c) for c in kwargs["count"]) #TODO: parallelize this
            return list(chain.from_iterable(state_dicts))
        state.synth = self.train_adapter.generate_data(
            model=state.model,
            **kwargs,
        )
        if self.output_format is not None:
            state.synth = self.ensure_output_format(state.synth)
        return [state]
