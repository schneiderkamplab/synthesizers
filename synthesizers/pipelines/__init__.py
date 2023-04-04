from typing import Any, Dict, Optional, Tuple

from .base import (
    PipelineRegistry,
)
from .tabular_synthesis import TabularSynthesisPipeline, TabularSynthesisDPPipeline

# Register all the supported tasks here
TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
    "vqa": "visual-question-answering",
}
SUPPORTED_TASKS = {
    "tabular-synthesis": {
        "impl": TabularSynthesisPipeline,
        "model": (),
        "default": None,
        "type": "tabular",
    },
    # This task is a special case as it's parametrized by EPSILON and DELTA.
    "tabular-synthesis-dp": {
        "impl": TabularSynthesisDPPipeline,
        "model": (),
        "default": None,
        "type": "tabular",
    },
}

PIPELINE_REGISTRY = PipelineRegistry(supported_tasks=SUPPORTED_TASKS, task_aliases=TASK_ALIASES)

def check_task(task: str) -> Tuple[str, Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - `"tabular-synthesis"`
            - `"tabular-synthesis_EPSILON_DELTA"`

    Returns:
        (normalized_task: `str`, task_defaults: `dict`, task_options: (`tuple`, None)) The normalized task name
        (removed alias and options). The actual dictionary required to initialize the pipeline and some extra task
        options for parametrized tasks like "tabular-synthesis_EPSILON_DELTA"


    """
    return PIPELINE_REGISTRY.check_task(task)

def pipeline(
    task: str = None,
    model: Optional[Any] = None,
    pipeline_class: Optional[Any] = None,
):
    normalized_task, targeted_task, task_options = check_task(task)
    if pipeline_class is None:
        pipeline_class = targeted_task["impl"]
