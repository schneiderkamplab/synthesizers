from datasets import load_dataset, DatasetDict
in_data_train = load_dataset("mstz/breast", split='train[:100]')
in_data = DatasetDict()
in_data['train'] = in_data_train
print(in_data)
from synthesizers import pipeline
p = pipeline("train")
model = p(in_data)
print(model)
p = pipeline("generate")
out_data = p(model, count=100)
print(out_data)
p = pipeline("synthesize")
out_data = p(in_data)
print(out_data)
p = pipeline("evaluate")
result = p(out_data['train'], out_data['generated'])
print(result)
