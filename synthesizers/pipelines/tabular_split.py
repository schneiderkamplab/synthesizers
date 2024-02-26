from collections.abc import Iterable
from itertools import chain
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import List, Optional, Union

from .base import Pipeline
from ..utils.formats import State, StateDict

class TabularSplitPipeline(Pipeline):

    def _call(
        self,
        state: StateDict,
        size: Optional[Union[int, float, List[Union[int, float]]]] = None,
    ) -> List[StateDict]:
        kwargs = dict(self.kwargs)
        kwargs.update(self.split_args)
        if size is None:
            if kwargs.get("size", None) is None:
                kwargs["size"] = 0.8
        else:
            kwargs["size"] = size
        if isinstance(kwargs["size"], Iterable) and not isinstance(kwargs["size"], str):
            state_dicts = (self._call(state.clone(), size=s) for s in kwargs["size"]) #TODO: parallelize this
            return list(chain.from_iterable(state_dicts))
        kwargs["train_size"] = kwargs["size"]
        del kwargs["size"]
        train = self.ensure_output_format(state.train, DataFrame)
        train, test = train_test_split(
            train,
            **kwargs,
        )
        state.train = train.reset_index(drop=True)
        state.test = test.reset_index(drop=True)
        if self.output_format is not None:
            state.train = self.ensure_output_format(state.train)
            state.test = self.ensure_output_format(state.test)
        return [state]
