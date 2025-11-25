from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import torch

# -----------------------------
# 1. Training Data (Example)
# -----------------------------
texts = [
    "I want to reset my password",
    "How can I change my plan?",
    "Payment failed yesterday",
    "The app is not opening",
    "Need help updating my profile",
    "I was double charged for my subscription",
    "App keeps crashing",
]

labels = [
    0,   # account_issue
    3,   # sales
    1,   # billing
    2,   # technical
    0,   # account_issue
    1,   # billing
    2,   # technical
]

label_names = ["account_issue", "billing", "technical", "sales"]

dataset = Dataset.from_dict({"text": texts, "label": labels})

# Split small dataset
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# -----------------------------
# 2. Load Tokenizer & Model
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

tokenized_ds = dataset.map(tokenize, batched=True)

# -----------------------------
# 3. Load DistilBERT model
# -----------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_names)
)

# -----------------------------
# 4. Training configuration
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    logging_first_step=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)



# -----------------------------
# 5. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
)

# -----------------------------
# 6. Train the model
# -----------------------------
trainer.train()

# -----------------------------
# 7. Save the model
# -----------------------------
model.save_pretrained("./distilbert_model")
tokenizer.save_pretrained("./distilbert_model")

print("Model training complete. Saved to ./distilbert_model")
