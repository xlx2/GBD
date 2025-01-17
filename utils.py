import numpy as np


def pow2dB(x):
    return 10 * np.log10(x)


def dB2pow(x):
    return 10 ** (x / 10)


def pow2dBm(x):
    return 10 * np.log10(x / 1e-3)


def dBm2pow(x):
    return 10 ** (x / 10) / 1e3


def eigenvalue_decomposition(xxH: np.ndarray) -> np.ndarray:
    """
    This function decomposes the semi-definite hermitian matrix xxH into a matrix x,
    where the eigenvectors and square roots of the eigenvalues. x @ xH = xxH
    :param xxH: Semi-definite hermitian matrix
    :return: x
    """
    if not isinstance(xxH, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if not np.allclose(xxH, np.conj(xxH.T)):
        raise ValueError("Input matrix must be Hermitian (symmetric if real-valued).")
    eigenvalues, eigenvectors = np.linalg.eig(xxH)
    if np.any(eigenvalues < 0):
        raise ValueError("Input matrix must be positive semi-definite.")
    x = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    return x


def create_boolean_vector(length_of_vector: int, num_of_ones: int) -> np.ndarray:
    """
    This function generates a boolean column vector of size N,
    containing num_of_ones random distributed 1, and the rest 0.
    :param length_of_vector: Length of the vector
    :param num_of_ones: Number of random ones
    :return: Boolean vector
    """
    if length_of_vector <= 0:
        raise ValueError("length_of_vector must be greater than 0.")
    if num_of_ones > length_of_vector:
        raise ValueError("num_of_ones must be less than or equal to N.")
    x = np.zeros((length_of_vector, 1))
    selected_indices = np.random.choice(length_of_vector, num_of_ones, replace=False)
    x[selected_indices, :] = 1
    return x


def create_block_diag_matrix(x: np.ndarray, repeat: int = None) -> np.ndarray:
    """
    This function creates a block diagonal matrix from an input x.
    1. If x is a column vector, it first repeats the vector to form a matrix with a shape of (row, repeat),
    then it repeats the matrix to form a block diagonal matrix.
    2. If x is a matrix, it directly forms the block diagonal matrix.
    :param x: Input 2D numpy ndarray
    :param repeat: Number of times to repeat the vector, and None for matrix
    :return: Block diagonal matrix
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if x.ndim != 2:
        raise TypeError("Input must be a 2D array.")

    row, column = x.shape
    if column == 1:  # X is a column vector
        if repeat is None:
            raise ValueError("`repeat` must be provided when the input is a column vector.")
        x = np.repeat(x, repeat, axis=1)  # Repeat the column vector to form a matrix with a shape of (row, repeat)
        column = repeat

    block_diag_matrix = np.zeros((column, row * column), dtype=np.complex128)

    for col in range(column):
        for r in range(row):
            block_diag_matrix[col, r + col * row] = x[r, col]

    return block_diag_matrix
