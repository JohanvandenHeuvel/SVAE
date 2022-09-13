import torch

from distributions.gaussian import info_to_standard
from matrix_ops import outer_product
from info_ops import info_marginalize


def info_rst_smoothing(J, h, cond_msg, pred_msg, pair_params, loc_next):
    J_cond, h_cond = cond_msg
    J_pred, h_pred = pred_msg
    J11, J12, J22 = pair_params

    temp = J - J_pred + J22
    # TODO make marginalize local?
    J_smooth, h_smooth = info_marginalize(
        A=temp, B=J12, C=(J_cond + J11), h1=h - h_pred, h2=h_cond
    )

    loc, scale = info_to_standard(J_smooth, h_smooth)
    E_xnxT = -torch.linalg.solve(temp, J12.T) @ scale + outer_product(loc_next, loc)
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
        return E_xxT, E_xnxT.T, E_xnxnT, torch.tensor([1.0], device=E_x.device)

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
    (J_smooth, h_smooth), _ = forward_messages[-1]
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
