from typing import Any, Dict, List, Optional, Tuple, Type, Union

from ..adapters import NAME_TO_ADAPTER
from ..adapters.base import Adapter
from ..utils import ensure_format
from ..utils import logging

logger = logging.get_logger(__name__)

class PipelineRegistry:
    def __init__(self, supported_tasks: Dict[str, Any], task_aliases: Dict[str, str]) -> None:
        self.supported_tasks = supported_tasks
        self.task_aliases = task_aliases

    def get_supported_tasks(self) -> List[str]:
        supported_task = list(self.supported_tasks.keys()) + list(self.task_aliases.keys())
        supported_task.sort()
        return supported_task

    def check_task(self, task: str) -> Tuple[str, Dict, Any]:
        if task in self.task_aliases:
            task = self.task_aliases[task]
        if task in self.supported_tasks:
            targeted_task = self.supported_tasks[task]
            return task, targeted_task, None

        if task.startswith("tabular-synthesis-dp"):
            tokens = task.split("_")
            if len(tokens) == 3 and tokens[0] == "tabular-synthesis-dp":
                targeted_task = self.supported_tasks["tabular-synthesis"]
                task = "tabular-synthesis"
                return task, targeted_task, (float(tokens[1]), float(tokens[2]))
            raise KeyError(f"Invalid tabular-synthesis-dp task {task}, use 'tabular-synthesis-dp_EPSILON_DELTA' format with EPSILON and DELTA being floats")

        raise KeyError(
            f"Unknown task {task}, available tasks are {self.get_supported_tasks() + ['tabular-synthesis-dp_EPSILON_DELTA']}"
        )

class Pipeline():
    def __init__(self,
            task: str,
            adapter: Optional[Union[Adapter, str]] = None,
            output_format: Optional[Type] = None,
            **kwargs,
        ):
        self.task = task
        if isinstance(adapter, str):
            adapter = NAME_TO_ADAPTER[adapter]()
        self.adapter = adapter
        self.output_format = output_format
        self.kwargs = kwargs
    def ensure_output_format(self, data, output_format=None, **kwargs):
        return ensure_format(data, (self.output_format if output_format is None else output_format,), **kwargs)
