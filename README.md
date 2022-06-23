# SVAE

Re-implementation of [Composing graphical models with neural networks for structured representations and fast inference](https://arxiv.org/abs/1603.06277), for my Master's thesis.

**Paper Abstract**: * We propose a general modeling and inference framework that composes probabilistic graphical models with deep learning methods and combines their respective strengths. Our model family augments graphical structure in latent variables with neural network observation models. For inference, we extend variational autoencoders to use graphical model approximating distributions with recognition networks that output conjugate potentials. All components of these models are learned simultaneously with a single objective, giving a scalable algorithm that leverages stochastic variational inference, natural gradients, graphical model message passing, and the reparameterization trick. We illustrate this framework with several example models and an application to mouse behavioral phenotyping. *


## Repository Structure

```bash
.
├── data.py                     # generate synthetic data 
├── dense.py                    # functions to put parameters in single object
├── distributions               # contains various distributions
│   ├── categorical.py     
│   ├── dirichlet.py
│   ├── distribution.py         # superclass for exponential distributions
│   ├── gaussian.py
│   └── niw.py
├── log.py                      # functions to help saving information
├── plot                        # plotting functionality
│   └── plot.py                 
├── README.md
├── svae                        # contains svae specific code 
│   ├── global_optimization.py  # code for optimizing global variables
│   ├── local_optimization.py   # code for optimizing local variables
│   ├── main.py                 # code to just run svae
│   └── model.py                # contains the svae algorithm
└── vae                         # contains generic variational autoencoder code
    ├── main.py                 # code to just run vae
    └── models
        ├── addmodule.py        # custom module for adding two functions
        ├── resvae.py           # standard vae with skip connections
        └── vae.py              # standard vae
```
