from collections.abc import Iterable
from itertools import chain
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from subprocessing import Pool
from typing import List, Optional, Union

from .base import Pipeline
from ..utils.formats import StateDict

class TabularSplitPipeline(Pipeline):

    def _call(
        self,
        state_dict: StateDict,
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
            if self.jobs is None:
                state_dicts = (self._call(state_dict.clone(), size=s) for s in kwargs["size"])
            else:
                with Pool(processes=self.jobs) as pool:
                    state_dicts = pool.map(self._call, [(state_dict.clone(),) for _ in kwargs["size"]], ({"size": s} for s in kwargs["size"]))
            return list(chain.from_iterable(state_dicts))
        kwargs["train_size"] = kwargs["size"]
        del kwargs["size"]
        train = self.ensure_output_format(state_dict.train, DataFrame)
        train, test = train_test_split(
            train,
            **kwargs,
        )
        state_dict.train = train.reset_index(drop=True)
        state_dict.test = test.reset_index(drop=True)
        if self.output_format is not None:
            state_dict.train = self.ensure_output_format(state_dict.train)
            state_dict.test = self.ensure_output_format(state_dict.test)
        return [state_dict]
