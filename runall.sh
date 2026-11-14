# Check that Turbo Boost is disabled
NO_TURBO_FILE="/sys/devices/system/cpu/intel_pstate/no_turbo"

if [ -r "$NO_TURBO_FILE" ] && [ "$(cat "$NO_TURBO_FILE")" = "1" ]; then
    echo "Turbo Boost is disabled."
else
    echo "Error: Turbo Boost must be disabled before the run." >&2
    exit 1
fi

sudo nice -n -15 taskset -c 0-7 sudo -u "$USER" -E zsh -c '
    . autoenv-enter.zsh &&
    cd exp/12-2026-11-13-vbe-bfgs-many-trials-jit-and-warmup &&
    exec zsh run.sh
'

sudo nice -n -15 taskset -c 0-7 sudo -u "$USER" -E zsh -c '
    . autoenv-enter.zsh &&
    cd exp/13-2026-11-13-vbe-bfgs-many-trials-jit-and-warmup-raw &&
    exec zsh run.sh
'
