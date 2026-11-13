export JAX_PLATFORMS=cpu

export OUTDIR_ROOT=_output
export SCRIPT=../../bin/train_viscous_burgers_pinn.py

mkdir -p "${OUTDIR_ROOT}"

RAW_SCIPY=raw_scipy_optimize

NTRIALS=42

for i in $(seq -f "%02g" ${NTRIALS}); do
    OUTDIR=${OUTDIR_ROOT}/trial_${i}
    mkdir "${OUTDIR}"

    # ---
    # Python SciPy.optimize directly, without open interfaces
    # with BFGS
    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    # ---
    # Python SciPy.optimize directly, without open interfaces
    # with L-BFGS
    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt L-BFGS --gtol 1e-3 --split-key --seed "${i}" >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt L-BFGS --gtol 1e-4 --split-key --seed "${i}" >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt L-BFGS --gtol 1e-5 --split-key --seed "${i}" >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"
done
