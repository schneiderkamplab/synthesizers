from collections.abc import Iterable
from itertools import chain
from subprocessing import Pool
from typing import List, Optional, Union

from ..adapters import *
from .base import Pipeline
from ..models.auto import MODEL_TO_ADAPTER
from ..utils.formats import State, StateDict

class TabularGenerationPipeline(Pipeline):

    def _call(
        self,
        state_dict: StateDict,
        count: Optional[Union[int, List[int]]] = None,
    ) -> List[StateDict]:
        kwargs = dict(self.kwargs)
        kwargs.update(self.gen_args)
        if count is None:
            if kwargs.get("count", None) is None:
                kwargs["count"] = 1 if state_dict.train is None else len(state_dict.train)
        else:
            kwargs["count"] = count
        if self.train_adapter is None:
            self.train_adapter = eval(MODEL_TO_ADAPTER[state_dict.model.__class__])()
        if isinstance(kwargs["count"], Iterable) and not isinstance(kwargs["count"], str):
            if self.jobs is None:
                state_dicts = (self._call(state_dict.clone(), count=c) for c in kwargs["count"])
            else:
                with Pool(processes=self.jobs) as pool:
                    state_dicts = pool.map(self._call, [(state_dict.clone(),) for _ in kwargs["count"]], ({"count": c} for c in kwargs["count"]))
            return list(chain.from_iterable(state_dicts))
        state_dict.synth = self.train_adapter.generate_data(
            model=state_dict.model,
            **kwargs,
        )
        if self.output_format is not None:
            state_dict.synth = self.ensure_output_format(state_dict.synth)
        return [state_dict]
