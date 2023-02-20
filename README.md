# SVAE

Re-implementation of [Composing graphical models with neural networks for structured representations and fast inference](https://arxiv.org/abs/1603.06277), for my Master's thesis. The Structured Variational Autoencoder (SVAE) can be used to combine probabilistic graphical models with non-linear observations. For more informations please look at msc_thesis.pdf

**Paper Abstract**: *We propose a general modeling and inference framework that composes probabilistic graphical models with deep learning methods and combines their respective strengths. Our model family augments graphical structure in latent variables with neural network observation models. For inference, we extend variational autoencoders to use graphical model approximating distributions with recognition networks that output conjugate potentials. All components of these models are learned simultaneously with a single objective, giving a scalable algorithm that leverages stochastic variational inference, natural gradients, graphical model message passing, and the reparameterization trick. We illustrate this framework with several example models and an application to mouse behavioral phenotyping.*

## repository structure
.
├── README.md
├── data.py				# code for generating synthetic data
├── distributions			# contains probability distributions
├── figures				# contains code for replicating thesis figures
├── log.py
├── matrix_ops.py
├── notebooks				
├── plot				# contains helper functions for plotting
├── run_gmm.py				# code to run SVAE-GMM
├── run_lds.py				# code to run SVAE-LDS
├── seed.py
├── svae				# contains the SVAE implementation
│   ├── gmm
│   │   ├── global_optimization.py
│   │   ├── local_optimization.py
│   │   └── model.py
│   ├── gradient.py
│   └── lds
│       ├── global_optimization.py
│       ├── kalman			# local optimization operations
│       │   ├── filter.py
│       │   ├── info_ops.py
│       │   ├── sample.py
│       │   └── smoothing.py
│       ├── local_optimization.py
│       └── model.py
├── trained_gmm
├── trained_lds
├── trained_vae
└── vae					# contains the VAE implementation
    ├── models
    │   ├── __init__.py
    │   ├── addmodule.py
    │   ├── autoencoder.py
    │   ├── resvae.py
    │   └── vae.py
    └── run_vae.py
