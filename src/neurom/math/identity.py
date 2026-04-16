import torch


def identity(shape, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Compute identity tensor

    The identity is computed over the field dimensions for all elements and quadrature points N_e an N_q.

    Args:
        shape (torch.Size): Shape of the field for which we want to compute the identity
    Returns:
        A torch.Tensor of shape (N_e, N_q, *field_shape) representing the identity of the field.
    """
    if len(shape) < 3:
        raise ValueError(f"Need at least 3 dimensions but got '{shape}'")

    n_e, n_q = shape[:2]
    dim = shape[2:]

    # Scalar case
    if dim == (1,):
        return torch.ones(n_e, n_q, 1, dtype=dtype, device=device)

    # Matrix field
    if len(dim) == 2 and dim[0] == dim[1]:
        d = dim[0]
        I = torch.eye(d, dtype=dtype, device=device)
        return I.expand(n_e, n_q, d, d)

    raise ValueError(
        f"Identity only defined for scalar (1,) or square matrix (d,d), got shape: '{shape}'"
    )
