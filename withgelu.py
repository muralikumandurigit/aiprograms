# python
import torch
import torch.nn as nn

vocab = {"good": 1, "bad": 0, "average": 2, "worst": 3, "excellent": 4}
train_data = {"Good Excellent Average": 1,
              "Bad Worst": 0,
              "Bad Average": 0,
              "Excellent Good": 1}
vocab_size = len(vocab)
embedding_dim = 8
hidden_dim = 16
output_dim = 2
num_heads = 8

def encode(text):
    tokens = text.lower().split()
    return [vocab[word] for word in tokens if word in vocab]

class MyMiniGleuModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        # x expected shape: (seq_len, batch)
        x = self.embedding(x)                 # -> (seq_len, batch, embed_dim)
        attn_out, _ = self.attention(x, x, x) # expects (L, N, E)
        x = x + attn_out
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = x.mean(dim=0)                     # -> (batch, embed_dim)
        x = self.classifier(x)                # -> (batch, output_dim)
        return x

model = MyMiniGleuModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    total_loss = 0
    for text, label in train_data.items():
        ids = encode(text)
        if len(ids) == 0:
            ids = [0]  # fallback index when text contains no known tokens
        inputs = torch.tensor(ids, dtype=torch.long).unsqueeze(1)  # (seq_len, batch=1)
        targets = torch.tensor([label], dtype=torch.long)          # (batch=1)

        optimizer.zero_grad()
        outputs = model(inputs)            # (batch, output_dim)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")

    # Testing
    test_texts = ["good great", "worst bad", "average"]
    for test_text in test_texts:
        ids = encode(test_text)
        if len(ids) == 0:
            ids = [0]
        test_input = torch.tensor(ids, dtype=torch.long).unsqueeze(1)
        with torch.no_grad():
            test_output = model(test_input)                # (batch, output_dim)
            predicted_class = torch.argmax(test_output, dim=1).item()
            print(f"Input: '{test_text}' => Predicted class: {predicted_class}")