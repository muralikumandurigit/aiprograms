import torch
import torch.nn as nn

# -----------------------------
# 1. Create synthetic data
# -----------------------------
# y = 3x + 2 (our ground truth)
x = torch.randn(10000, 1)
y = 3 * x + 2

# -----------------------------
# 2. Define the model
# -----------------------------
model = nn.Linear(1, 1)   # single linear layer

# -----------------------------
# 3. Loss function & optimizer
# -----------------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# -----------------------------
# 4. Training loop
# -----------------------------
for epoch in range(20000):
    pred = model(x)              # forward pass
    loss = loss_fn(pred, y)      # compute loss

    optimizer.zero_grad()        # clear old gradients
    loss.backward()              # compute gradients
    optimizer.step()             # update weights

    if epoch % 2000 == 0:
        print(f"Epoch {epoch} | Loss = {loss.item():.6f}")

print("\nTraining finished.\n")

# -----------------------------
# 5. TESTING:
# Compare actual vs predicted
# -----------------------------

# Pick 5 random test points
test_x = torch.tensor([[1.0], [2.0], [-1.0], [0.5], [-3.0]])
true_y = 3 * test_x + 2        # ground truth
pred_y = model(test_x)         # model prediction

print("Test Input (x):")
print(test_x)

print("\nActual y = 3x + 2:")
print(true_y)

print("\nPredicted y by model:")
print(pred_y.detach())

# -----------------------------
# 6. Print learned parameters
# -----------------------------
weight, bias = list(model.parameters())
print("\nLearned weight:", weight.item())
print("Learned bias:", bias.item())
