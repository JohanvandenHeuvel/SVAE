import torch

from info_ops import info_marginalize


def lognorm(J, h, full=False):
    n = len(h)
    constant = n * torch.log(torch.tensor(2 * torch.pi)) if full else 0.0
    return 0.5 * (h @ torch.linalg.solve(J, h) - torch.slogdet(J)[1] + constant)


def info_condition(J, h, J_obs, h_obs):
    # if not torch.all(torch.linalg.eigvalsh(J) >= 0.0):
    #     raise ValueError(f"J not pd: {torch.linalg.eigvalsh(J)}")
    return J + J_obs, h + h_obs


def info_predict(J, h, J11, J12, J22, logZ):
    new_J = J + J11
    # TODO make marginalize local?
    J_pred, h_pred = info_marginalize(A=new_J, B=J12.T, C=J22, h1=h, h2=0)
    logZ_pred = 0.5 * (
        h @ torch.linalg.solve(new_J, h) - torch.linalg.slogdet(new_J)[1]
    )
    return J_pred, h_pred, logZ_pred.squeeze() + logZ


def info_kalman_filter(init_params, pair_params, observations):
    (J, h), logZ = init_params
    J11, J12, J22, logZ_param = pair_params

    if not torch.all(torch.linalg.eigvalsh(J) >= 0.0):
        raise ValueError(f"init J not pd: {torch.linalg.eigvalsh(J)}")

    total_logZ = logZ
    forward_messages = []
    for i, (J_obs, h_obs) in enumerate(observations):
        J_cond, h_cond = info_condition(J, h, J_obs, h_obs)
        J, h, logZ = info_predict(J_cond, h_cond, J11, J12, J22, logZ_param)
        total_logZ += logZ.squeeze()
        forward_messages.append(((J_cond, h_cond), (J, h)))
    total_logZ += lognorm(J, h)
    return forward_messages, total_logZ
