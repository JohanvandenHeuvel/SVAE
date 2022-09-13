import torch

from matrix_ops import symmetrize


def info_observation_params(obs, C, R):
    R_inv = torch.inverse(R)
    R_inv_C = R_inv @ C

    # J_obs = C.T @ inv(R) @ C
    J_obs = C.T @ R_inv_C
    # h_obs = (y - D @ u) @ inv(R) @ C
    h_obs = obs @ R_inv_C

    J_obs = J_obs.unsqueeze(0).repeat(len(obs), 1, 1)
    return zip(J_obs, h_obs)


def info_pair_params(A, Q):
    J22 = torch.inverse(Q)
    J12 = -A.T @ J22
    J11 = A.T @ -J12
    return J11, J12, J22


def info_marginalize(A, B, C, h1, h2):
    # TODO make marginalize local?
    J = symmetrize(C - B @ torch.linalg.solve(A, B.T))
    h = h2 - B @ torch.linalg.solve(symmetrize(A), h1)
    return J, h.squeeze()
