import numpy as np

def pack_dense(A, b, c=None, d=None):
    """

    For example:

    A = [[a11, a12],
        [a21, a22]]
    b = [b1, b2]
    and c and d are scalars.

    Then the result is:

    [[a11, a12, b1, 0],
    [a21,  a22, b2, 0],
    [0,      0,  c, 0],
    [0,      0,  0, d]]

    Parameters
    ----------
    A :
        Scale matrix.
    b :
        loc vector.
    c :
        scalar
    d :
        scalar

    Returns
    -------

    """

    leading_dim = b.shape[:-1]
    last_dim = b.shape[-1]

    z1 = np.zeros(leading_dim + (last_dim, 1))
    z2 = np.zeros(leading_dim + (1, 1))

    if c is None:
        c = z2
    if d is None:
        d = z2

    b = b[..., None]

    c = np.reshape(c, leading_dim + (1, 1))
    d = np.reshape(d, leading_dim + (1, 1))

    return np.concatenate((
        np.concatenate((A, b, z1), axis=-1),
        np.concatenate((np.swapaxes(z1, axis1=-1, axis2=-2), c, z2), axis=-1),
        np.concatenate((np.swapaxes(z1, axis1=-1, axis2=-2), z2, d), axis=-1),
    ), axis=-2)

def unpack_dense(arr):
    N = arr.shape[-1] - 2
    return arr[..., :N, :N], arr[..., :N, N], arr[..., N, N], arr[..., N+1, N+1]
