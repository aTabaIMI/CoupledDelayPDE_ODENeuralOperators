import numpy as np
from scipy.integrate import quad
def solve_system(
    gamma=5,
    K=-2,
    A=1,
    B=1,
    C=1,
    N=50,
    num_iterations=5,
):
    x = np.linspace(0, 1, N)
    h = 1.0 / N
    def lambda_func(z):
        eps = 1e-12
        z = np.clip(z, -1 + eps, 1 - eps)
        return np.sin((gamma + 1) * np.arccos(z)) / np.sqrt(1 - z**2)

    # ------------------------------------------------------------------
    # Initial gamma0(x)
    # ------------------------------------------------------------------
    def gamma0(z):
        def inner_integral(eta):
            result, _ = quad(lambda xi: lambda_func(xi) * C, 0, eta)
            return result

        outer_result, _ = quad(inner_integral, 0, z)
        return K - outer_result
        # ------------------------------------------------------------------
    # Compute gamma(x) iteratively
    # ------------------------------------------------------------------
    def compute_gamma(x_vals):
        gamma_vals = np.zeros((num_iterations, len(x_vals)))
        gamma_vals[0, :] = np.array([gamma0(z) for z in x_vals])

        for it in range(1, num_iterations):
            for idx, z in enumerate(x_vals):
                integral1, _ = quad(
                    lambda eta: quad(
                        lambda xi: np.interp(
                            xi, x_vals, gamma_vals[it - 1, :]
                        )
                        * A,
                        0,
                        eta,
                    )[0],
                    0,
                    z,
                )

                integral2, _ = quad(
                    lambda y: quad(
                        lambda xi: ((z - y - xi) ** 2)
                        * np.interp(
                            xi, x_vals, gamma_vals[it - 1, :]
                        )
                        * B
                        * lambda_func(y)
                        * C,
                        0,
                        z - y,
                    )[0],
                    0,
                    z,
                )

                gamma_vals[it, idx] = integral1 + 0.5 * integral2

        return np.sum(gamma_vals, axis=0)
    f = compute_gamma(x)
    # ------------------------------------------------------------------
    # Finite difference solution U
    # ------------------------------------------------------------------
    U = np.zeros((N + 2, N + 2))

    # Boundary condition: U[i, i] = 0
    for i in range(1, N + 1):
        U[i, i] = 0.0

    # Initialization
    U[2, 1] = h * f[2] * B
    U[3, 1] = 2 * U[2, 1] - U[1, 1] + h * (f[3] - f[0]) * B
    U[3, 2] = U[3, 1] - h * f[3] * B

    # Interior update
    for i in range(4, N + 1):
        for j in range(2, i - 2):
            U[i, j] = (
                U[i - 1, j + 1]
                + U[i - 1, j - 1]
                - U[i - 2, j]
            )

        U[i, 1] = U[i, 2] + h * f[3] * B
        U[i, i - 1] = U[i, i - 2] - U[i - 1, i - 2]

    return U ,x
