from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np

model_path = "./deberta_multilabel_massive/model"
peft_config = PeftConfig.from_pretrained(model_path)
base_model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=8,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True
)
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

text = "Dear Support, I am quite frustrated with the slow response time of your application. It takes too long to load, and sometimes it does not respond at all. This has affected my ability to complete tasks efficiently. I hope you can address these performance issues as soon as possible."
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits[0].numpy()

LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
for label, logit in zip(LABELS, logits):
    print(f"{label}: {logit:.4f}")

