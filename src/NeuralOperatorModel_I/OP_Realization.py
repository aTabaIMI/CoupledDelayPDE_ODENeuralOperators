from FD_Solver import *

def generate_realizations(n_realizations=100, seed=1234):
    """
    Create independent parameter sets.
    """
    rng = np.random.default_rng(seed)

    realizations = []

    for k in range(n_realizations):
        params = {
            'id': k,
            'gamma': rng.uniform(2.0, 8.0),
            'K': rng.uniform(-5.0, 5.0),
            'A': 1.0,
            'B': 1.0,
            'C': 1.0,
        }
        realizations.append(params)

    return realizations

def split_realizations(realizations, train_ratio=0.8, seed=1234):
    """
    Split entire realizations into train and test sets.
    """
    rng = np.random.default_rng(seed)

    indices = np.arange(len(realizations))
    rng.shuffle(indices)

    n_train = int(train_ratio * len(indices))

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_realizations = [realizations[i] for i in train_indices]
    test_realizations = [realizations[i] for i in test_indices]

    return train_realizations, test_realizations

def extract_samples_from_solution(U, x_grid, realization_id, mx=50, my=50):
    """
    Convert one numerical solution into DeepONet samples.

    Returns
    -------
    us : branch inputs
    xs : trunk inputs
    ss : targets
    ids : realization identifiers
    """

    us, xs, ss, ids = [], [], [], []

    y_lists = [np.linspace(0.0, xi, my) for xi in x_grid]

    for i in range(mx):
        xi = x_grid[i]

        for j, yi in enumerate(y_lists[i]):
            if yi <= xi:

                # Branch input (same representation as in your original script)
                u_branch = U[:, j].astype(np.float32)

                # Trunk input
                x_trunk = np.array([xi, yi], dtype=np.float32)

                # Target value
                s_target = np.array([U[i, j]], dtype=np.float32)

                us.append(u_branch)
                xs.append(x_trunk)
                ss.append(s_target)
                ids.append(realization_id)

    return (
        np.asarray(us, dtype=np.float32),
        np.asarray(xs, dtype=np.float32),
        np.asarray(ss, dtype=np.float32),
        np.asarray(ids, dtype=np.int64),
    )

def build_dataset(realization_list, mx=50, my=50):
    """
    Solve each realization and concatenate all samples.
    """

    all_us = []
    all_xs = []
    all_ss = []
    all_ids = []

    for params in realization_list:
        print(f"Solving realization {params['id']}...");

        U, x_grid = solve_system(
            gamma=params['gamma'],
            K=params['K'],
            A=params['A'],
            B=params['B'],
            C=params['C']
        )

        us, xs, ss, ids = extract_samples_from_solution(
            U=U,
            x_grid=x_grid,
            realization_id=params['id'],
            mx=50,
            my=50,
        )

        all_us.append(us)
        all_xs.append(xs)
        all_ss.append(ss)
        all_ids.append(ids)

    return (
        np.vstack(all_us),
        np.vstack(all_xs),
        np.vstack(all_ss),
        np.concatenate(all_ids),
    )

def verify_no_operator_leakage(train_ids, test_ids):
    """
    Ensure no realization identifier appears in both sets.
    """

    train_set = set(train_ids.tolist())
    test_set = set(test_ids.tolist())

    overlap = train_set.intersection(test_set)

    if len(overlap) > 0:
        raise RuntimeError(
            f"Operator leakage detected. Overlapping IDs: {sorted(overlap)}"
        )

    print("✓ No operator leakage detected.")
    print(f"  Number of train realizations: {len(train_set)}")
    print(f"  Number of test realizations : {len(test_set)}")

import torch
from torch.utils.data import TensorDataset, DataLoader


def make_dataloader(us, xs, ss, batch_size=256, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(us, dtype=torch.float32),
        torch.tensor(xs, dtype=torch.float32),
        torch.tensor(ss, dtype=torch.float32),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

mx = 50
nx = 50
# ---------------------------------------------------------
# 1. Generate many independent operator realizations
# ---------------------------------------------------------
realizations = generate_realizations(
    n_realizations=100,
    seed=1234,
)

# ---------------------------------------------------------
# 2. Split by realization (NOT by samples)
# ---------------------------------------------------------
train_realizations, test_realizations = split_realizations(
    realizations,
    train_ratio=0.8,
    seed=1234,
)

# ---------------------------------------------------------
# 3. Build training dataset
# ---------------------------------------------------------
train_us, train_xs, train_ss, train_ids = build_dataset(
    train_realizations,
    mx=50,
    my=50,
)

# ---------------------------------------------------------
# 4. Build testing dataset
# ---------------------------------------------------------
test_us, test_xs, test_ss, test_ids = build_dataset(
    test_realizations,
    mx=50,
    my=50,
)

# ---------------------------------------------------------
# 5. Verify zero operator overlap
# ---------------------------------------------------------
verify_no_operator_leakage(train_ids, test_ids)

# ---------------------------------------------------------
# 6. Create DataLoaders
# ---------------------------------------------------------
train_loader = make_dataloader(
    train_us,
    train_xs,
    train_ss,
    batch_size=256,
    shuffle=True,
)

test_loader = make_dataloader(
    test_us,
    test_xs,
    test_ss,
    batch_size=256,
    shuffle=False,
)

