export JAX_PLATFORMS=cpu

export OUTDIR_ROOT=_output
export SCRIPT=../../bin/train_viscous_burgers_pinn.py

echo "USER = ${USER}"
echo "python = $(which python)"

mkdir -p "${OUTDIR_ROOT}"

OIF_SCIPY=scipy_optimize
RAW_SCIPY=raw_scipy_optimize

NTRIALS=42

for i in $(seq -f "%02g" ${NTRIALS}); do
    OUTDIR=${OUTDIR_ROOT}/trial_${i}
    mkdir "${OUTDIR}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OIF_SCIPY} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt BFGS --gtol 1e-3 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OIF_SCIPY} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt BFGS --gtol 1e-4 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${OIF_SCIPY} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"

    python ${SCRIPT} --outdir="${OUTDIR}" --impl ${RAW_SCIPY} --opt BFGS --gtol 1e-5 --split-key --seed "${i}" --jit --warmup >log.log 2>&1 
    outdir_local=$(grep "Experiment" log.log | cut -d" " -f4)
    mv log.log "${OUTDIR}/${outdir_local}"
done
