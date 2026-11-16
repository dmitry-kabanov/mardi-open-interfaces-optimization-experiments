# Just for sanity output the user and the actual Python used.
echo "USER = ${USER}"
echo "python = $(which python)"

# Use CPU for JAX computations
export JAX_PLATFORMS=cpu

# The path to the script that runs actual computations.
export SCRIPT=../../bin/train_viscous_burgers_pinn.py

export OUTDIR_ROOT=_output
mkdir -p "${OUTDIR_ROOT}"

SCIPY=scipy_optimize
OPTIM=optim_jl
RAW_SCIPY=raw_scipy_optimize

# Sleep $TIME seconds between running different simulations
# to minimize the effect of CPU thermals.
TIME=180

# We run multiple trials to report averaged results.
NTRIALS=42

# -----------------------------------------------------------------------------
# Create trial directories
for i in $(seq -f "%02g" ${NTRIALS}); do
    OUTDIR=${OUTDIR_ROOT}/trial_${i}
    mkdir "${OUTDIR}"
done

# -----------------------------------------------------------------------------
# Python SciPy.optimize via MaRDI Open Interfaces
for i in $(seq -f "%02g" ${NTRIALS}); do
    OUTDIR=${OUTDIR_ROOT}/trial_${i}

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${SCIPY} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${SCIPY} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${SCIPY} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"
done

sleep $TIME

# -----------------------------------------------------------------------------
# Julia Optim.jl
for i in $(seq -f "%02g" ${NTRIALS}); do
    OUTDIR=${OUTDIR_ROOT}/trial_${i}

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" --linesearch HagerZhang --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" --linesearch HagerZhang --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" --linesearch HagerZhang --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"
done

sleep $TIME

# -----------------------------------------------------------------------------
# Julia Optim.jl using linesearch=StrongWolfe
for i in $(seq -f "%02g" ${NTRIALS}); do
    OUTDIR=${OUTDIR_ROOT}/trial_${i}

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" --linesearch StrongWolfe --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" --linesearch StrongWolfe --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OPTIM} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" --linesearch StrongWolfe --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"
done

sleep $TIME

# -----------------------------------------------------------------------------
# Julia Optim.jl using linesearch=BackTracking
for i in $(seq -f "%02g" ${NTRIALS}); do
    OUTDIR=${OUTDIR_ROOT}/trial_${i}

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

sleep $TIME

# -----------------------------------------------------------------------------
# Python SciPy.optimize directly, without open interfaces
# with BFGS
for i in $(seq -f "%02g" ${NTRIALS}); do
    OUTDIR=${OUTDIR_ROOT}/trial_${i}

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"
done

sleep $TIME

# -----------------------------------------------------------------------------
# Python SciPy.optimize directly, without open interfaces
# with L-BFGS
for i in $(seq -f "%02g" ${NTRIALS}); do
    OUTDIR=${OUTDIR_ROOT}/trial_${i}

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt L-BFGS --gtol 1e-3 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt L-BFGS --gtol 1e-4 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt L-BFGS --gtol 1e-5 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"
done
