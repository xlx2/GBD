import numpy as np


def pow2dB(x):
    return 10 * np.log10(x)


def dB2pow(x):
    return 10 ** (x / 10)


def pow2dBm(x):
    return 10 * np.log10(x / 1e-3)


def dBm2pow(x):
    return 10 ** (x / 10) / 1e3


def eigenvalue_decomposition(XX_H) -> np.ndarray:
    """
    This function decomposes the semi-definite hermitian matrix XX_H into a
    matrix X, where the eigenvectors and square roots of the eigenvalues.
    X @ X_H = XX_H

    :param XX_H: Semi-definite hermitian matrix
    :return: X @ X_H = XX_H
    """
    if not isinstance(XX_H, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    eigenvalues, eigenvectors = np.linalg.eig(XX_H)
    X = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    return X


def get_boolean_vector(N: int, numOfOnes: int) -> np.ndarray:
    """
    This function generates a boolean vector of size N,
    containing numOfOnes random distributed 1, and the rest 0.
    :param N: Length of the vector
    :param numOfOnes: Number of random 1s
    :return: Boolean vector of size N
    """
    if N <= 0:
        raise ValueError("N must be greater than 0.")
    if numOfOnes > N:
        raise ValueError("numOfOnes must be less than or equal to N.")
    x = np.zeros((N, 1))
    selected_indices = np.random.choice(N, numOfOnes, replace=False)
    x[selected_indices, :] = 1
    return x


def create_block_diag_matrix(X: np.ndarray) -> np.ndarray:
    """
    Constructs a block diagonal matrix from the columns of the input matrix X.
    Each column of H is placed as the diagonal elements of a block in the resulting matrix.

    :param X: A 2D numpy array of shape (N, K), where N is the number of rows and
              K is the number of columns.
    :return: A 2D numpy array of shape (N, N * K), representing the block diagonal matrix.
             The diagonal blocks correspond to the columns of H.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input H must be a numpy ndarray.")
    if X.ndim != 2:
        raise ValueError("Input H must be a 2D array.")

    N, K = X.shape  # Extract dimensions of the input matrix
    block_diag_matrix = np.zeros((N, N * K), dtype=np.complex128)  # Initialize the output block diagonal matrix

    for k in range(K):
        # Place each column of H as the diagonal block in the output matrix
        block_diag_matrix[:, k * N:(k + 1) * N] = np.diag(X[:, k])

    return block_diag_matrix


