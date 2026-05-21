import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Parameters ---
N = 120
h = 1.0 / N
x_grid = np.linspace(0, 1, N + 1)
y_grid = np.linspace(0, 1, N + 1)

# Constants
A = 1
B = 1
C = 1
K = -2
gamma = 5

# ------------------------------------------------------------
# Define lambda and gamma0 functions
# ------------------------------------------------------------
def lambda_func(x):
    return np.sin((gamma + 1) * np.arccos(x)) / np.sqrt(1 - x**2 + 1e-10)


def gamma0(x):
    def inner_integral(eta):
        result, _ = quad(lambda xi: lambda_func(xi) * C, 0, eta)
        return result

    outer_result, _ = quad(inner_integral, 0, x)
    return K - outer_result


def compute_gamma(x_vals):
    gamma_vals = np.zeros((5, len(x_vals)))
    gamma_vals[0, :] = np.array([gamma0(x) for x in x_vals])

    for i in range(1, 5):
        for idx, x in enumerate(x_vals):
            integral1, _ = quad(
                lambda eta: quad(
                    lambda xi: np.interp(
                        xi, x_vals, gamma_vals[i - 1, :]
                    ) * A,
                    0,
                    eta,
                )[0],
                0,
                x,
            )

            integral2, _ = quad(
                lambda y: quad(
                    lambda xi: ((x - y - xi) ** 2)
                    * np.interp(xi, x_vals, gamma_vals[i - 1, :])
                    * B
                    * lambda_func(y)
                    * C,
                    0,
                    x - y,
                )[0],
                0,
                x,
            )

            gamma_vals[i, idx] = integral1 + 0.5 * integral2

    return np.sum(gamma_vals, axis=0)
