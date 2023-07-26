from typing import Any, Dict, List, Optional, Tuple, Union

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

    def register_pipeline(
        self,
        task: str,
        pipeline_class: type,
        pt_model: Optional[Union[type, Tuple[type]]] = None,
        tf_model: Optional[Union[type, Tuple[type]]] = None,
        default: Optional[Dict] = None,
        type: Optional[str] = None,
    ) -> None:
        if task in self.supported_tasks:
            logger.warning(f"{task} is already registered. Overwriting pipeline for task {task}...")

        if pt_model is None:
            pt_model = ()
        elif not isinstance(pt_model, tuple):
            pt_model = (pt_model,)

        if tf_model is None:
            tf_model = ()
        elif not isinstance(tf_model, tuple):
            tf_model = (tf_model,)

        task_impl = {"impl": pipeline_class, "pt": pt_model, "tf": tf_model}

        if default is not None:
            if "model" not in default and ("pt" in default or "tf" in default):
                default = {"model": default}
            task_impl["default"] = default

        if type is not None:
            task_impl["type"] = type

        self.supported_tasks[task] = task_impl
        pipeline_class._registered_impl = {task: task_impl}

    def to_dict(self):
        return self.supported_tasks

class Pipeline():
    def __init__(self, task, adapter):
        self.task = task
        self.adapter = adapter
