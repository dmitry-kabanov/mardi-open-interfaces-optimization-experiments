export JAX_PLATFORMS=cpu

export OUTDIR_ROOT=_output
export SCRIPT=../../bin/train_viscous_burgers_pinn.py

mkdir -p "${OUTDIR_ROOT}"

SCIPY=scipy_optimize
OPTIM=optim_jl

NTRIALS=42

for i in $(seq -f "%02g" ${NTRIALS}); do
    OUTDIR=${OUTDIR_ROOT}/trial_${i}
    mkdir "${OUTDIR}"

    # ---
    # Python SciPy.optimize
    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${SCIPY} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${SCIPY} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${SCIPY} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    # ---
    # Julia Optim.jl
    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" --linesearch HagerZhang --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" --linesearch HagerZhang --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" --linesearch HagerZhang --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    # ---
    # Julia Optim.jl using linesearch=StrongWolfe for fairer comparison with SciPy
    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" --linesearch StrongWolfe --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" --linesearch StrongWolfe --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" --linesearch StrongWolfe --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    # ---
    # Julia Optim.jl using linesearch=BackTracking for fairer comparison with SciPy
    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" --linesearch BackTracking --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" --linesearch BackTracking --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" --linesearch BackTracking --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

done
