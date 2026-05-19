import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
from FD_Solver import *
from OP_Realization import *

M = 50
N = 50

class DeepONet(nn.Module):
    def __init__(self, neurons=20, in1=N+2, in2=2, output_neurons=10):  # ✅ Fix here
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(in1, 2 * neurons), nn.ReLU(),
            nn.Linear(2 * neurons, neurons), nn.ReLU(),
            nn.Linear(neurons, output_neurons)
        )
        self.trunk = nn.Sequential(
            nn.Linear(in2, neurons), nn.ReLU(),
            nn.Linear(neurons, neurons), nn.ReLU(),
            nn.Linear(neurons, output_neurons)
        )

    def forward(self, x1, x2):
        return torch.einsum('bi,bi->b', self.branch(x1), self.trunk(x2)).unsqueeze(1)

model = DeepONet(in1=N + 2, in2=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


train_losses, test_losses = [], []



def closure():
    optimizer.zero_grad()
    total_loss = 0.0
    batch_count = 0
    for u, x, s in train_loader:
        pred = model(u, x)
        mse_loss = torch.norm(pred - s)**2 / s.numel()  # MSE loss
        mse_loss.backward()
        total_loss += mse_loss.item()
        batch_count += 1
    avg_loss = total_loss / batch_count
    train_losses.append(avg_loss)
    return total_loss


for epoch in range(50):
    optimizer.step(closure)

    # ارزیابی روی داده تست
    model.eval()
    total_test_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for u, x, s in test_loader:
            pred = model(u, x)
            mse_loss = torch.norm(pred - s)**2 / s.numel()  # MSE loss
            total_test_loss += mse_loss.item()
            batch_count += 1
    avg_test_loss = total_test_loss / batch_count
    test_losses.append(avg_test_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch+1}, Train MSE Loss: {train_losses[-1]:.6f}, Test MSE Loss: {test_losses[-1]:.6f}")

        plt.figure()  # Create a new 2D figure and reset current axes
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss", linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Relative Loss")
plt.title("DeepONet Training - S(x, y)")
plt.legend()
plt.grid(True)
plt.show()


from mpl_toolkits.mplot3d import Axes3D


x_pred = np.linspace(0, 1, N)
y_pred_list = [np.linspace(0, xi, N) for xi in x_pred]

U,x = solve_system( gamma=5,
    K=-2,
    A=1,
    B=1,
    C=1,
    N=50,
    num_iterations=5,
)
M = N
X_pred, Y_pred, U_pred = [], [], []

model.eval()
with torch.no_grad():
    for i, xi in enumerate(x_pred):
        for yi in y_pred_list[i]:
            
            u_input = torch.tensor(U[:, int(yi * (M - 1))], dtype=torch.float32).unsqueeze(0)
            x_input = torch.tensor([[xi, yi]], dtype=torch.float32)

            # پیش‌بینی
            s_pred = model(u_input, x_input).item()
            X_pred.append(xi)
            Y_pred.append(yi)
            U_pred.append(s_pred)



fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X_pred, Y_pred, U_pred, cmap='viridis', edgecolor='none')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Predicted S(x, y)")
ax.set_title("DeepONet Prediction Surface for S(x, y)")
plt.show()

torch.save(model.state_dict(), "deeponet_trained.pth")
