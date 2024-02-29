print("IMPORTING LIBRARIES")
from datasets import load_dataset
import synthcity.metrics.eval_statistical
import synthcity.plugins
from synthesizers import pipeline
from synthesizers.models import AutoModel
from synthesizers.adapters import SynthPopAdapter
from synthesizers.utils.formats import ensure_format, SUPPORTED_FORMATS, State, StateDict

def lim_print(data, limit=800):
    res = repr(data).replace("\n","")
    res = res if len(res) <= limit else res[:limit-3]+"..."
    print(type(data), res)

print("LOADING TEST DATASET")
in_data = load_dataset("mstz/breast", split='train[:100]')
print("TESTING FORMAT CONVERSIONS")
data = {}
for format in SUPPORTED_FORMATS:
    tmp_data = ensure_format(in_data, target_formats=(format,))
    assert type(tmp_data) == format
    lim_print(tmp_data)
    data[format] = tmp_data
for source_format in SUPPORTED_FORMATS:
    for target_format in SUPPORTED_FORMATS:
        tmp_data = ensure_format(data[source_format], target_formats=(target_format,))
        assert type(tmp_data) == target_format
        lim_print(tmp_data)
        assert ensure_format(tmp_data, target_formats=(list,)) == data[list]

print("TESTING TRAINING")
p = pipeline("train")
model = p(in_data)
lim_print(model)
model[0].model.save_pretrained("test/synthcity_model")
p = pipeline("train", train_adapter="synthpop")
model = p(in_data)
lim_print(model)
model[0].model.save_pretrained("test/synthpop_model")

print("TESTING GENERATION")
model = AutoModel.from_pretrained("test/synthcity_model")
state = StateDict(model=model)
p = pipeline("generate", jobs=1)
out_data = p(state, count=100)
lim_print(out_data)
model = AutoModel.from_pretrained("test/synthpop_model")
state = StateDict(model=model)
p = pipeline("generate")
out_data = p(state, count=100)[0].synth
lim_print(out_data)

print("TESTING SYNTHEVAL")
p = pipeline("evaluate", eval_adapter="syntheval", target_col="is_cancer")
state = StateDict(train=in_data, synth=out_data)
result = p(state)[0].eval
lim_print(result)
p = pipeline("evaluate", eval_adapter="syntheval")
result = p(state)[0].eval
lim_print(result)
p = pipeline("evaluate", eval_adapter="syntheval", config="fast_eval")
result = p(state)[0].eval
lim_print(result)
p = pipeline("evaluate", eval_adapter="syntheval", target_col="is_cancer")
result = p(state)[0].eval
lim_print(result)

print("TESTING SYNTHESIS")
p = pipeline("synthesize")
out_data = p(in_data)[0].synth
lim_print(out_data)
p = pipeline("synthesize", train_adapter=SynthPopAdapter())
out_data = p(in_data)[0].synth
lim_print(out_data)

print("TESTING EVALUATION")
print("default", end=": ")
p = pipeline("evaluate")
state = StateDict(train=in_data, synth=out_data)
result = p(state)
lim_print(result)
for value in synthcity.metrics.eval_statistical.__dict__.values():
    try:
        if value.type() == "stats":
            p = pipeline("evaluate", evaluator_class=value)
            result = p(state)
            print(value.name(), end=": ")
            lim_print(result)
    except:
        pass

print("TESTING SYNTHCITY PLUGINS")
for plugin in synthcity.plugins.Plugins().list():
    try:
        p = pipeline("train", plugin=plugin)
        model = p(in_data)
        lim_print(model)
    except:
        pass