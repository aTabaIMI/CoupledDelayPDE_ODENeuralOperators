from AnalyticalFunctions import *
from DE_solver import *

def build_dataset(P_data, x_vals, y_vals, u_boundary_norm):
    us, xs, ps = [], [], []

    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            us.append(u_boundary_norm)
            xs.append([x_vals[i], y_vals[j]])
            ps.append(P_data[i, j])

    return (
        np.array(us),
        np.array(xs),
        np.array(ps).reshape(-1, 1),
    )


us_P, xs_P, ps_P = build_dataset(P_norm, x, y, u_norm)

dataset = TensorDataset(
    torch.tensor(us_P, dtype=torch.float32),
    torch.tensor(xs_P, dtype=torch.float32),
    torch.tensor(ps_P, dtype=torch.float32),
)

# ============================================================
# LEAK-FREE TRAIN / TEST SPLIT
# ============================================================

num_x = len(x)
num_y = len(y)

train_num_x = int(0.8 * num_x)

# Indices corresponding to flattened ordering:
# index = i * num_y + j
train_indices = []
test_indices = []

for i in range(num_x):
    for j in range(num_y):
        idx = i * num_y + j
        if i < train_num_x:
            train_indices.append(idx)
        else:
            test_indices.append(idx)

train_data = Subset(dataset, train_indices)
test_data = Subset(dataset, test_indices)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False
)

print(f"Total samples: {len(dataset)}")
print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")
print(f"Training x-range: [0, {x[train_num_x-1]:.4f}]")
print(f"Testing x-range:  [{x[train_num_x]:.4f}, 1]")
