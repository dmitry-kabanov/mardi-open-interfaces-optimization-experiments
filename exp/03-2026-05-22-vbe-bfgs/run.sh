export JAX_PLATFORMS=cpu

export OUTDIR=exp/03-2026-05-22-vbe-bfgs

SCIPY=scipy_optimize
OPTIM=optim_jl

# ---
# Python SciPy.optimize

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${SCIPY} --opt BFGS --gtol 1e-3 >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${SCIPY} --opt BFGS --gtol 1e-4 >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${SCIPY} --opt BFGS --gtol 1e-5 >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${SCIPY} --opt BFGS --gtol 1e-3 --split-key >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${SCIPY} --opt BFGS --gtol 1e-4 --split-key >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${SCIPY} --opt BFGS --gtol 1e-5 --split-key >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

# ---
# Julia Optim.jl

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-3 >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-4 >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-5 >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-3 --split-key >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-4 --split-key >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"

python bin/train_viscous_burgers_pinn.py --outdir=${OUTDIR} --impl ${OPTIM} --opt BFGS --gtol 1e-5 --split-key >log.log 2>&1 
outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
mv log.log "${OUTDIR}/${outdir_local}"
