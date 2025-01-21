import numpy as np


class FluidAntennaSystem:
    def __init__(
        self,
        num_of_yaxis_antennas: int,
        num_of_users: int,
        noise_variance: float,
        num_of_xaxis_antennas: int = 1,
        Wx=1,
        Wy=1
    ):
        self.Nx = num_of_xaxis_antennas
        self.Ny = num_of_yaxis_antennas
        if self.Ny == 1:
            raise ValueError("If you want a linear antenna array,"
                             " please set num_of_xaxis_antennas to one instead.\n")
        self.K = num_of_users
        self.sigma2 = noise_variance
        self.Wx = Wx
        self.Wy = Wy
        self.N = self.Nx * self.Ny

        # Create 2D indices
        ntx2, ntx1 = np.meshgrid(range(1, self.Ny + 1), range(1, self.Nx + 1))
        self.ntx1 = ntx1.ravel()
        self.ntx2 = ntx2.ravel()

        # Compute spatial correlation matrix
        self.J, self.dist1, self.dist2, self.d = self._compute_spatial_correlation()

        # Eigenvalue decomposition
        Ltx, self.Utx = np.linalg.eig(self.J)
        self.Ltx = np.diag(Ltx)

    def _compute_spatial_correlation(self):
        d1 = np.zeros((self.N, self.N))
        d2 = np.zeros((self.N, self.N))

        for i in range(self.N):
            if (self.Nx - 1) != 0:
                d1[:, i] = np.abs(self.ntx1[i] - self.ntx1) / (self.Nx - 1) * self.Wx
            d2[:, i] = np.abs(self.ntx2[i] - self.ntx2) / (self.Ny - 1) * self.Wy

        d = np.sqrt(d1 ** 2 + d2 ** 2)  # Total distance
        J = self.sigma2 * np.sinc(2 * np.pi * d)  # Spatial correlation
        J[np.isnan(J)] = 1  # Replace NaN with 1 (for zero distance)

        return J, d1, d2, d

    def get_channel(self):
        g = np.sqrt(0.5) * (np.random.randn(self.N, self.K) + 1j * np.random.randn(self.N, self.K))
        h = np.zeros((self.N, self.K), dtype=complex)

        for k in range(self.K):
            h[:, k:k+1] = self.Utx @ self.Ltx.T @ g[:, k:k+1]

        return h
