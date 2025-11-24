import json
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# ---------- LOAD DATA ----------
with open("training_data.json", "r") as f:
    data = json.load(f)

texts = [d["text"] for d in data]
labels = [d["label"] for d in data]

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.4, random_state=42, stratify=labels
)

# ---------- TOKENIZER ----------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_batch(text_list, label_list):
    return tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    ), torch.tensor(label_list)

train_encodings, train_labels = encode_batch(X_train, y_train)
test_encodings, test_labels = encode_batch(X_test, y_test)

# ---------- DATASET CLASS ----------
class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_ds = TicketDataset(train_encodings, train_labels)
test_ds = TicketDataset(test_encodings, test_labels)

# ---------- MODEL ----------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

# ---------- TRAINING SETTINGS ----------
training_args = TrainingArguments(
    output_dir="./bert_model",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    logging_dir='./logs',
    logging_steps=10
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

print("Training starting...")
trainer.train()
print("Training completed!")

# Save model + labels
trainer.save_model("model_bert")
import joblib
joblib.dump(label_encoder, "model_bert/label_encoder.pkl")
print("Model saved successfully.")
