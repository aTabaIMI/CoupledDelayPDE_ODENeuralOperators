from AnalyticalFunctions import *
# ------------------------------------------------------------
# Compute initial velocity f
# ------------------------------------------------------------
f = compute_gamma(x_grid)

# ------------------------------------------------------------
# First PDE: kernel U
# ------------------------------------------------------------
U = np.zeros((N + 2, N + 2))

for i in range(1, N + 1):
    U[i, i] = 0

U[2, 1] = h * f[2] * B
U[3, 1] = 2 * U[2, 1] - U[1, 1] + h * (f[3] - f[0]) * B
U[3, 2] = U[3, 1] - h * f[3] * B

for i in range(4, N + 1):
    for j in range(2, i - 2):
        U[i, j] = U[i - 1, j + 1] + U[i - 1, j - 1] - U[i - 2, j]

    # NOTE: keeping your original formula intentionally
    U[i, 1] = U[i, 2] + h * f[3] * B
    U[i, i - 1] = U[i, i - 2] - U[i - 1, i - 2]

# Extract k(1,y)
U_1_y = U[N, : N + 1]
t_u = y_grid

# ------------------------------------------------------------
# Second PDE: P_x = D P_yy
# ------------------------------------------------------------
D = 2.0
my_u = N + 1
dy = 1 / (my_u - 1)

dx = np.sqrt(2 * D * dy)
mx = int(1 / dx) + 1
dx = 1 / (mx - 1)

r2 = D * (dy / dx**2)

x = np.linspace(0, 1, mx)
y = np.linspace(0, 1, my_u)

P = np.zeros((mx, my_u))
P[0, :] = np.interp(y, t_u, U_1_y)
P[:, 0] = 0
P[:, -1] = 0

for i in range(mx - 1):
    for j in range(1, my_u - 1):
        P[i + 1, j] = (
            P[i, j]
            + r2 * (P[i, j + 1] - 2 * P[i, j] + P[i, j - 1])
        )

# No normalization
u_norm = U_1_y
P_norm = P
