from AnalyticalFunctions import *
from DE_solver import *
from Dataset import *

# ============================================================
# DeepONet Model
# ============================================================
class DeepONet(nn.Module):
    def __init__(self, m):
        super().__init__()

        self.branch = nn.Sequential(
            nn.Linear(m, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, u, x):
        branch_out = self.branch(u)
        trunk_out = self.trunk(x)
        return self.fc(branch_out * trunk_out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepONet(m=len(u_norm)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================
# Relative L2 Error
# ============================================================
def relative_error(pred, target, eps=1e-12):
    return torch.norm(pred - target) / (torch.norm(target) + eps)

# ============================================================
# Training
# ============================================================
epochs = 1000
train_losses = []
test_losses = []

for epoch in range(epochs):
    # --------------------
    # Training
    # --------------------
    model.train()
    running_rel_loss = 0.0

    for u_batch, x_batch, p_batch in train_loader:
        u_batch = u_batch.to(device)
        x_batch = x_batch.to(device)
        p_batch = p_batch.to(device)

        optimizer.zero_grad()

        pred = model(u_batch, x_batch)
        rel_loss = relative_error(pred, p_batch)

        rel_loss.backward()
        optimizer.step()

        running_rel_loss += rel_loss.item()

    train_losses.append(running_rel_loss / len(train_loader))

    # --------------------
    # Testing
    # --------------------
    model.eval()
    test_rel_loss = 0.0

    with torch.no_grad():
        for u_batch, x_batch, p_batch in test_loader:
            u_batch = u_batch.to(device)
            x_batch = x_batch.to(device)
            p_batch = p_batch.to(device)

            pred = model(u_batch, x_batch)
            rel_loss = relative_error(pred, p_batch)

            test_rel_loss += rel_loss.item()

    test_losses.append(test_rel_loss / len(test_loader))

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch+1:4d} | "
            f"Train Rel Error: {train_losses[-1]:.6e} | "
            f"Test Rel Error: {test_losses[-1]:.6e}"
        )

# ============================================================
# Prediction on the whole domain
# ============================================================
model.eval()

P_pred = np.zeros_like(P_norm)

with torch.no_grad():
    u_input = torch.tensor(
        u_norm, dtype=torch.float32
    ).unsqueeze(0).to(device)

    for i in range(len(x)):
        for j in range(len(y)):
            x_input = torch.tensor(
                [[x[i], y[j]]],
                dtype=torch.float32
            ).to(device)

            P_pred[i, j] = model(u_input, x_input).item()

# ============================================================
# Surface plots
# ============================================================
X, Y = np.meshgrid(y, x)

fig = plt.figure(figsize=(14, 6))

# Numerical solution
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(X, Y, P, cmap="viridis")
ax1.set_title("P(x,y) - Numerical Solution")
ax1.set_xlabel("y")
ax1.set_ylabel("x")
ax1.set_zlabel("P")

# DeepONet prediction
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot_surface(X, Y, P_pred, cmap="viridis")
ax2.set_title("P(x,y) - DeepONet Prediction")
ax2.set_xlabel("y")
ax2.set_ylabel("x")
ax2.set_zlabel("P_pred")

plt.tight_layout()
plt.show()

# ============================================================
# Training / Testing Relative Error
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Relative Error")
plt.plot(test_losses, label="Test Relative Error")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Relative Error")
plt.title("Leak-Free Train/Test Relative Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Global relative error over the full domain
# ============================================================
global_rel_error = (
    np.linalg.norm(P_pred - P) /
    (np.linalg.norm(P) + 1e-12)
)

print(f"\nGlobal Relative L2 Error = {global_rel_error:.6e}")
