import torch

from distributions import MatrixNormalInverseWishart, NormalInverseWishart
from distributions.dense import pack_dense
from distributions.gaussian import (
    info_to_standard,
    Gaussian,
    info_to_natural,
    natural_to_info,
)

device = "cuda:0"


def symmetrize(A):
    return (A + A.T) / 2.0


def is_posdef(A):
    return torch.allclose(A, A.T) and torch.all(torch.linalg.eigvalsh(A) >= 0.0)


def outer_product(x, y):
    # computes xyT
    return torch.einsum("i, j -> ij", (x, y))


def info_condition(J, h, J_obs, h_obs):
    assert torch.allclose(J, J.T)
    assert torch.all(torch.linalg.eigvalsh(J) >= 0.0)

    assert torch.allclose(J_obs, J_obs.T)
    assert torch.all(torch.linalg.eigvalsh(J_obs) >= 0.0)

    assert torch.allclose((J + J_obs), (J + J_obs).T)
    assert torch.all(torch.linalg.eigvalsh((J + J_obs)) >= 0.0)

    return J + J_obs, h + h_obs


def condition(J, h, y, Jxx, Jxy):
    J_cond = J + Jxx
    h_cond = h - (Jxy @ y.T).T
    return J_cond, h_cond


def lognorm(J, h, full=False):
    n = len(h)
    constant = n * torch.log(torch.tensor(2 * torch.pi)) if full else 0.0
    return 0.5 * (h @ torch.linalg.solve(J, h) + torch.slogdet(J)[1] + constant)


def info_marginalize(J11, J12, J22, h, logZ):
    # # assert logZ < 0
    # # J11_inv = torch.inverse(J11)
    # # temp = J12.T @ J11_inv
    # temp = torch.linalg.solve(J11, J12)
    #
    # # J_pred = J22 - J12.T @ inv(J11) @ J12
    # J_pred = symmetrize(J22 - temp @ J12)
    # # h_pred = h2 - J12.T @ inv(J11) @ h1
    # h_pred = -temp @ h
    # logZ_pred = logZ - 1/2 h1.T @ inv(J11) @ h1 + 1/2 log|J11| - n/2 log(2pi)
    # logZ_pred = logZ - lognorm(J11, h)

    ###################
    #     CHOL        #
    ###################

    L = torch.linalg.cholesky(J11)
    v = torch.linalg.solve_triangular(L, h[..., None], upper=False)

    h_pred = -J12.T @ torch.linalg.solve_triangular(L.T, v, upper=False)
    temp = torch.linalg.solve_triangular(L, J12, upper=False)
    J_pred = J22 - temp.T @ temp

    logZ_pred = logZ - 0.5 * v.T @ v - torch.sum(torch.log(torch.diag(L)))

    if not is_posdef(J_pred):
        raise ValueError("Predicted matrix is not positive-definite")

    return J_pred, h_pred.squeeze(), logZ_pred.squeeze()


def info_predict(J, h, J11, J12, J22, logZ):
    J_new = J + J11
    return info_marginalize(J_new, J12, J22, h, logZ)


def info_kalman_filter(init_params, pair_params, observations):
    J, h = init_params
    J11, J12, J22, logZ = pair_params

    forward_messages = []
    for i, (J_obs, h_obs) in enumerate(observations):
        J_cond, h_cond = info_condition(J, h, J_obs, h_obs)
        J, h, logZ = info_predict(J_cond, h_cond, J11, J12, J22, logZ)
        forward_messages.append(((J_cond, h_cond), (J, h)))

    return forward_messages, logZ


def info_rst_smoothing(J, h, cond_msg, pred_msg, pair_params, loc_next):
    J_cond, h_cond = cond_msg
    J_pred, h_pred = pred_msg
    J11, J12, J22 = pair_params

    temp = J12 @ torch.inverse(J - J_pred + J22)
    J_smooth = J_cond + J11 - temp @ J12.T
    h_smooth = h_cond - temp @ (h - h_pred)

    loc, scale = info_to_standard(J_smooth, h_smooth)
    E_xnxT = temp @ scale + outer_product(loc_next, loc)
    E_xxT = scale + outer_product(loc, loc)

    stats = (loc, E_xxT, E_xnxT)

    return J_smooth, h_smooth, stats


def process_expected_stats(expected_stats):
    def make_init_stats(a):
        E_x, E_xxT, _ = a
        return (
            E_xxT,
            E_x,
            torch.tensor(1.0, device=E_x.device),
            torch.tensor(1.0, device=E_x.device),
        )

    def make_pair_stats(a, b):
        E_x, E_xxT, E_xnxT = a
        E_xn, E_xnxnT, _ = b
        # return E_xxT, E_xnxT.T, E_xnxnT, 1.0
        return E_xnxnT, E_xnxT.T, E_xxT, 1.0

    def make_node_stats(a):
        E_x, E_xxT, _ = a
        return torch.diag(E_xxT), E_x, 1.0

    E_init_stats = make_init_stats(expected_stats[0])
    E_pair_stats = [
        make_pair_stats(a, b) for a, b in zip(expected_stats[:-1], expected_stats[1:])
    ]
    # same pair for every time step
    E_pair_stats = [sum(stats) for stats in list(zip(*E_pair_stats))]

    E_node_stats = [make_node_stats(a) for a in expected_stats]
    E_node_stats = list(zip(*E_node_stats))
    E_node_stats[:2] = [torch.stack(E_stats) for E_stats in E_node_stats[:2]]
    E_node_stats[-1] = torch.tensor(E_node_stats[-1], device=E_node_stats[0].device)

    return E_init_stats, E_pair_stats, E_node_stats


def info_kalman_smoothing(forward_messages, pair_params):
    _, (J_smooth, h_smooth) = forward_messages[-1]
    loc, scale = info_to_standard(J_smooth, h_smooth)
    E_xxT = scale + outer_product(loc, loc)
    E_xnxT = 0.0

    expected_stats = [(loc, E_xxT, E_xnxT)]
    backward_messages = [(J_smooth, h_smooth)]
    for i, (cond_msg, pred_msg) in enumerate(reversed(forward_messages[:-1])):
        loc_next, _, _ = expected_stats[i]
        J_smooth, h_smooth, stats = info_rst_smoothing(
            J_smooth, h_smooth, cond_msg, pred_msg, pair_params, loc_next
        )
        backward_messages.append((J_smooth, h_smooth))
        expected_stats.append(stats)

    expected_stats = process_expected_stats(list(reversed(expected_stats)))

    return list(reversed(backward_messages)), expected_stats


def info_sample_backward(forward_messages, pair_params):
    J11, J12, _ = pair_params

    _, (J_pred, h_pred) = forward_messages[-1]
    next_sample = Gaussian(info_to_natural(J_pred, h_pred)).rsample()

    samples = [next_sample]

    means = [h_pred]
    variances = [-2 * J_pred]
    for _, (J_pred, h_pred) in reversed(forward_messages[:-1]):
        J = J_pred + J11
        h = h_pred - next_sample @ J12.T

        # get the sample
        state = Gaussian(info_to_natural(J, h.squeeze(0)))
        next_sample = state.rsample()
        samples.append(next_sample)

        means.append(h.squeeze(0))
        variances.append(-2 * J)

    samples = torch.stack(list(reversed(samples)))
    variances = torch.stack(list(reversed(variances)))
    means = torch.stack(list(reversed(means)))

    return samples, (means, variances)


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


def sample_forward_messages(messages):
    samples = []
    for _, (J, h) in messages:
        loc, scale = info_to_standard(J, h)
        x = loc + scale @ torch.randn(1, device=device)
        samples.append(x.cpu().detach().numpy())
    return samples


def sample_backward_messages(messages):
    samples = []
    for (J, h) in messages:
        loc, scale = info_to_standard(J, h)
        x = loc + scale @ torch.randn(1, device=device)
        samples.append(x.cpu().detach().numpy())
    return samples


def local_optimization(potentials, eta_theta):
    y = list(zip(*natural_to_info(potentials)))

    """
    priors 
    """
    niw_param, mniw_param = eta_theta

    J11, J12, J22, logZ = MatrixNormalInverseWishart(mniw_param).expected_stats()
    J11 *= -2
    J12 *= -1
    J22 *= -2

    """
    optimize local parameters
    """
    init_param = natural_to_info(NormalInverseWishart(niw_param).expected_stats())
    init_param = tuple([p.squeeze() for p in init_param])

    forward_messages, logZ = info_kalman_filter(
        init_params=init_param, pair_params=(J11, J12, J22, logZ), observations=y
    )
    backward_messages, expected_stats = info_kalman_smoothing(
        forward_messages, pair_params=(J11, J12, J22)
    )

    samples, (means, variances) = info_sample_backward(
        forward_messages, pair_params=(J11, J12, J22)
    )

    E_init_stats, E_pair_stats, E_node_stats = expected_stats
    local_kld = (
        torch.tensordot(
            potentials, pack_dense(E_node_stats[0], E_node_stats[1]), dims=3
        )
        - logZ
    )

    return samples, (E_init_stats, E_pair_stats), local_kld, (means, variances)
