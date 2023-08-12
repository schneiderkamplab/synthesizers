from .synthcity import SynthCityAdapter
from .syntheval import SynthEvalAdapter
from .synthpop import SynthPopAdapter

NAME_TO_ADAPTER = {
    "synthcity": SynthCityAdapter,
    "synthpop": SynthPopAdapter,
    "syntheval": SynthEvalAdapter,
}