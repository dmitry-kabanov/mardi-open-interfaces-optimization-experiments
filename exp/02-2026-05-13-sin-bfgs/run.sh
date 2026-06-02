#!/usr/bin/env zsh

python bin/train_sin_nn.py --opt BFGS --outdir=_output/exp02 --impl=scipy_optimize
python bin/train_sin_nn.py --opt BFGS --outdir=_output/exp02 --impl=optim_jl
