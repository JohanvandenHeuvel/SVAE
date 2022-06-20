import torch


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
    densely packed array
    """

    leading_dim = b.shape[:-1]
    last_dim = b.shape[-1]

    if A.ndim == b.ndim:
        """
        Turns e.g. 
        [a1, a2]
        into
        [a1, 0]
        [0, a2] 
        """
        A = A[..., None] * torch.eye(last_dim)[None, ...]

    # top rows where A and b go
    z1 = torch.zeros(leading_dim + (last_dim, 1))
    # bottom rows where c and d go
    z2 = torch.zeros(leading_dim + (1, 1))

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
