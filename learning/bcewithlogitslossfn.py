import torch
import torch.nn as nn

# -----------------------------------------
# 1. Create sample binary classification data
# -----------------------------------------
# x > 0 => label = 1
# x <= 0 => label = 0

x = torch.randn(100, 1)                 # 100 random numbers
y = (x > 0).float()                     # convert True/False → 1.0/0.0

# -----------------------------------------
# 2. Define a simple model
# -----------------------------------------
model = nn.Linear(1, 1)                 # single neuron

# -----------------------------------------
# 3. Loss & Optimizer
# -----------------------------------------
loss_fn = nn.BCEWithLogitsLoss()        # best for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# -----------------------------------------
# 4. Training loop
# -----------------------------------------
for epoch in range(200):
    logits = model(x)                   # raw output (logits)
    loss = loss_fn(logits, y)           # compute BCE loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss = {loss.item():.6f}")

print("\nTraining complete.\n")

# -----------------------------------------
# 5. Testing the model
# -----------------------------------------
test_x = torch.tensor([[-1.0], [0.5], [3.0], [-2.0]])
logits = model(test_x)
pred_prob = torch.sigmoid(logits)       # convert logits → probabilities
pred_class = (pred_prob > 0.5).float()  # threshold at 0.5

print("Test Input:")
print(test_x)

print("\nPredicted Probabilities:")
print(pred_prob.detach())

print("\nPredicted Class (0 or 1):")
print(pred_class)

print("\nActual Class (Ground truth):")
print((test_x > 0).float())
