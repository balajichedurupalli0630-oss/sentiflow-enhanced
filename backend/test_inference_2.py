import torch
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification

model_path = "./deberta_multilabel_massive/model"
peft_config = PeftConfig.from_pretrained(model_path)
base_model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path, num_labels=8, ignore_mismatched_sizes=True
)
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

texts = [
    "I am extremely angry and furious about this app! It crashes all the time!",
    "This app is complete garbage and I hate it.",
    "I am so sad and depressed right now."
]
for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    logits = model(**inputs).logits[0].detach().numpy()
    print(text)
    LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
    for label, logit in zip(LABELS, logits):
        print(f"  {label}: {logit:.4f}")
    print()

