from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import Optional, Union

from .base import Pipeline
from ..utils.formats import StateDict

class TabularSplitPipeline(Pipeline):
    def __call__(
        self,
        state: StateDict,
        size: Optional[Union[int, float]] = None,
    ):
        state = StateDict.wrap(state)
        kwargs = dict(self.kwargs)
        kwargs.update(self.split_args)
        if size is None:
            if kwargs.get("size", None) is None:
                kwargs["size"] = 0.8
        else:
            kwargs["size"] = size
        train = self.ensure_output_format(state.train, DataFrame)
        state.train, state.test = train_test_split(train, train_size=kwargs["size"], shuffle=True, random_state=42)
        if self.output_format is not None:
            state.train = self.ensure_output_format(state.train)
            state.test = self.ensure_output_format(state.test)
        return state
