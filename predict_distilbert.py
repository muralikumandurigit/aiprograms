from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

label_names = ["account_issue", "billing", "technical", "sales"]

model = DistilBertForSequenceClassification.from_pretrained("./distilbert_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert_model")

def classify(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
        prediction = output.logits.argmax(dim=1).item()
    return label_names[prediction]

# Test
print(classify("My payment didn’t go through"))
print(classify("I want to change my password"))
print(classify("App crashes when I open it"))
print(classify("How do I upgrade the subscription?"))
