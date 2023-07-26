from datasets import load_dataset
in_data = load_dataset("mstz/breast")
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
