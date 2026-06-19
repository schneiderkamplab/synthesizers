from .synthcity import SynthCityAdapter, SynthCityMetricsAdapter
from .syntheval import SynthEvalAdapter
from .synthpop import SynthPopAdapter

__all__ = [
    "NAME_TO_ADAPTER",
    "SynthCityAdapter",
    "SynthCityMetricsAdapter",
    "SynthEvalAdapter",
    "SynthPopAdapter",
]

NAME_TO_ADAPTER = {
    "synthcity": SynthCityAdapter,
    "synthcity-metrics": SynthCityMetricsAdapter,
    "synthpop": SynthPopAdapter,
    "syntheval": SynthEvalAdapter,
}
