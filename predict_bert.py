import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# Load model & tokenizer
model = BertForSequenceClassification.from_pretrained("model_bert")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
label_encoder = joblib.load("model_bert/label_encoder.pkl")

def classify_ticket(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    label = label_encoder.inverse_transform([predicted_id])[0]

    return label

# -------- TEST --------
print(classify_ticket("my bill is wrong"))
print(classify_ticket("internet is slow"))
print(classify_ticket("upgrade my plan"))
