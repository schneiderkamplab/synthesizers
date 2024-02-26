from collections.abc import Iterable
from itertools import chain
from typing import List, Optional, Union

from .base import Pipeline
from ..utils.formats import State, StateDict

class TabularTrainingPipeline(Pipeline):

    def _call(
        self,
        state: StateDict,
        plugin: Optional[Union[str, List[str]]] = None,
    ) -> List[StateDict]:
        kwargs = dict(self.kwargs)
        kwargs.update(self.train_args)
        if plugin is not None:
            kwargs["plugin"] = plugin
        if "plugin" in kwargs and isinstance(kwargs["plugin"], Iterable):
            state_dicts = (self._call(state.clone(), plugin=p) for p in kwargs["plugin"]) #TODO: parallelize this
            return list(chain.from_iterable(state_dicts))
        state.model = self.train_adapter.train_model(
            data=state.train,
            **kwargs,
        )
        return [state]
