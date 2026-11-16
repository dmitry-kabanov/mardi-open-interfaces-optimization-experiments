#!/usr/bin/env zsh

# Check that Turbo Boost is disabled
NO_TURBO_FILE="/sys/devices/system/cpu/intel_pstate/no_turbo"
if [ -r "$NO_TURBO_FILE" ]; then
    if [ "$(cat "$NO_TURBO_FILE")" = "1" ]; then
        echo "Turbo Boost is disabled. Proceeding"
    else
        echo "Error: Turbo Boost must be disabled before the run. Stopping" >&2
        exit 1
    fi
fi

# Check that $USER has permission to set `nice -15` (high priority)
if (ulimit -e >/dev/null 2>&1); then
    if [ "$(ulimit -e)" = "35" ]; then
        echo "User '$USER' can start processes with 'nice -n -15'. Proceeding"
    else
        echo "User '$USER' cannot start processes with 'nice -n -15'. Stopping" >&2
        exit 1
    fi
fi

cd exp/15-2026-11-15-vbe-all-together &&
    nice -n -15 taskset -c 0-7 zsh run.sh
