# Computational experiments with the `optim` interface of MaRDI Open Interfaces

This repository contains computational experiments
to illustrate the usage of the `optim` interface
of the [_MaRDI Open Interfaces_](https://github.com/MaRDI4NFDI/open-interfaces)
software package.


## Structure of the repository

```
├── bin                                                  # Training scripts
│   ├── train_sin_nn.py
│   └── train_viscous_burgers_pinn.py
├── exp                                                  # Experiments
│   ├── 01-2026-05-05-mlp-sine-func
│   ├── 02-2026-05-13-sin-bfgs
│   ├── 03-2026-05-22-vbe-bfgs
│   ├── 04-2026-05-26-vbe-bfgs-RUN2
│   ├── 05-2026-05-29-vbe-bfgs-optim_jl_with_StrongWolfe
│   ├── 06-2026-06-01-vbe-bfgs-compare-linesearches
│   └── 07-2026-06-02-vbe-bfgs-many-trials
├── README.md
├── src
│   └── mlpexample                                       # Common source code
└── tests/                                               # Tests
```

## Setup

It is assumed that _MaRDI Open Interfaces_ are installed
and available, i.e., dependencies are installed
and the environment variables are set.

Additionally, the following environment variables should be updated:
```shell
export PYTHONPATH="<repo-root>/src/mlpexample:${PYTHONPATH}"
export PATH="<repo-root>/bin:${PATH}"
```

## License

&copy; 2026 Dmitry I. Kabanov

The code in these repo is available under the CC BY 4.0 license,
see the license text
[here](https://creativecommons.org/licenses/by/4.0/legalcode.en).
