import pickle
import torch

from matrix_ops import pack_dense
from svae.lds.global_optimization import initialize_global_lds_parameters
from svae.lds.local_optimization import local_optimization

import numpy as np
np.set_printoptions(edgeitems=30, linewidth=100000)


potentials = pickle.load(open("potentials_999.pickle", "rb"), encoding="latin1")
mniw_param = pickle.load(open("mniw_param_999.pickle", "rb"), encoding="latin1")
mniw_param = list([torch.tensor(d).to("cuda:0").double() for d in mniw_param])

niw_prior, _ = initialize_global_lds_parameters(10)

J, h, _ = potentials
potentials = pack_dense(torch.tensor(J).to("cuda:0"), torch.tensor(h).to("cuda:0"))

x, (E_init_stats, E_pair_stats), local_kld = local_optimization(
    potentials, (niw_prior, mniw_param), n_samples=10
)

for arr in E_pair_stats:
    print(arr.cpu().detach().numpy())

