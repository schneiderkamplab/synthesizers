from collections.abc import Iterable
from itertools import chain
from subprocessing import Pool
from typing import List, Optional, Union

from .base import Pipeline
from ..utils.formats import StateDict

class TabularTrainingPipeline(Pipeline):

    def _call(
        self,
        state_dict: StateDict,
        plugin: Optional[Union[str, List[str]]] = None,
    ) -> List[StateDict]:
        kwargs = dict(self.kwargs)
        kwargs.update(self.train_args)
        if plugin is not None:
            kwargs["plugin"] = plugin
        if "plugin" in kwargs and isinstance(kwargs["plugin"], Iterable) and not isinstance(kwargs["plugin"], str):
            if self.jobs is None:
                state_dicts = (self._call(state_dict.clone(), plugin=p) for p in kwargs["plugin"])
            else:
                with Pool(processes=self.jobs) as pool:
                    state_dicts = pool.map(self._call, [(state_dict.clone(),) for _ in kwargs["plugin"]], ({"plugin": p} for p in kwargs["plugin"]))
            return list(chain.from_iterable(state_dicts))
        state_dict.model = self.train_adapter.train_model(
            data=state_dict.train,
            **kwargs,
        )
        return [state_dict]
