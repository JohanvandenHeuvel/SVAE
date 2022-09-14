import torch


def batch_diagonalize(A):
    return torch.diag_embed(A, dim1=-2, dim2=-1)


def batch_diagonal(A):
    return torch.diagonal(A, dim1=-2, dim2=-1)


def pack_dense(A, b, c=None, d=None):
    """

    For example for 2-Dimensional:

        A = [a11, a12]
            [a21, a22]
        b = [b1, b2]
        and c and d are scalars.

        Then the result is:
        [a11,  a12, b1, 0]
        [a21,  a22, b2, 0]
        [0,    0,   c,  0]
        [0,    0,   0,  d]

    Parameters
    ----------
    A:
        matrix.
    b:
        vector.
    c:
        scalar
    d:
        scalar

    Returns
    -------
    densely packed array
    """
    device = A.device

    leading_dim = b.shape[:-1]
    last_dim = b.shape[-1]

    if A.ndim == b.ndim:  # i.e. A is diagonal of diagonal matrix.
        A = batch_diagonalize(A)

    # top rows where A and b go
    z1 = torch.zeros(leading_dim + (last_dim, 1), device=device)
    # bottom rows where c and d go
    z2 = torch.zeros(leading_dim + (1, 1), device=device)

    if c is None:
        c = z2

    if d is None:
        d = z2

    c = torch.reshape(c, leading_dim + (1, 1))
    d = torch.reshape(d, leading_dim + (1, 1))

    b = b[..., None]

    return torch.cat(
        (
            torch.cat((A, b, z1), dim=-1),
            torch.cat((torch.swapaxes(z1, axis0=-1, axis1=-2), c, z2), dim=-1),
            torch.cat((torch.swapaxes(z1, axis0=-1, axis1=-2), z2, d), dim=-1),
        ),
        dim=-2,
    )


def unpack_dense(arr):
    """
    Inverts pack_dense

    Parameters
    ----------
    arr:
        dense array to unpack

    Returns
    -------
        A, b, c, d
    """
    N = arr.shape[-1] - 2
    return arr[..., :N, :N], arr[..., :N, N], arr[..., N, N], arr[..., N + 1, N + 1]


def is_posdef(A):
    """
    Check if matrix is positive definite. Raises ValueError if not.

    Parameters
    ----------
    A:
        Matrix.

    Returns
    -------

    """
    if not torch.allclose(A, A.T, atol=1e-6):
        raise ValueError(f"Matrix is not symmetric: \n {A.cpu().detach().numpy()}")

    eigenvalues = torch.linalg.eigvalsh(A)
    if not torch.all(eigenvalues >= 0.0):
        raise ValueError(f"Not all eigenvalues are positive: {eigenvalues}")

    return True


def batch_outer_product(x, y):
    """Computes xyT.

    e.g. if x.shape = (15, 2) and y.shape = (15, 2)
    then we get that first element of result equals [[x_0 * y_0, x_0 * y_1], [x_1 * y_0, x_1 * y_1]]

    """
    return torch.einsum("bi, bj -> bij", (x, y))


def batch_elementwise_multiplication(x, y):
    """Computes x * y where the fist dimension is the batch, x is a scalar.

    e.g. x.shape = (15, 1), y.shape = (15, 2, 2)
    then we get that first element of result equals x[0] * y[0]

    """
    assert x.shape[1] == 1
    return torch.einsum("ba, bij -> bij", (x, y))


def batch_matrix_vector_product(A, b):
    """Computes Ab for batch"""
    return torch.einsum("bij, bj -> bi", (A, b))


def outer_product(x, y):
    # computes xyT
    return torch.einsum("i, j -> ij", (x, y))


def symmetrize(A):
    T = lambda X: torch.swapaxes(X, axis0=-1, axis1=-2)
    return (A + T(A)) / 2
