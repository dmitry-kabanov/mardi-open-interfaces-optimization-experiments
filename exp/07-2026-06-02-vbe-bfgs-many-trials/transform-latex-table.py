import re
import sys


def adjust_uncertainty(match):
    mean_val = match.group(1)
    mean_exp = int(match.group(2))
    std_val = float(match.group(3))
    std_exp = int(match.group(4))

    # Scale standard deviation to match the mean's exponent
    adjusted_std = std_val * (10 ** (std_exp - mean_exp))

    # Return formatted string wrapped in \num{} with adjusted std rounded to 1 decimal
    return f"\\num{{{mean_val} +- {adjusted_std:.1f}e{match.group(2)}}}"


def _rename_index(label_: str) -> str:
    label = label_.strip()
    if label == "" or label == "{impl}":
        return label_

    if label == r"\multirow[t]{3}{*}{scipy_optimize}":
        return r"\multirow[c]{3}{*}{\parbox{1.24cm}{SciPy\\Opti-\\mize}}"
    elif label == r"\multirow[t]{3}{*}{optim_jl-HagerZhang}":
        return r"\multirow[c]{3}{*}{\parbox{1.24cm}{Optim.jl\\Hager\\Zhang}}"
    elif label == r"\multirow[t]{3}{*}{optim_jl-StrongWolfe}":
        return r"\multirow[c]{3}{*}{\parbox{1.24cm}{Optim.jl\\Strong\\Wolfe}}"
    elif label == r"\multirow[t]{3}{*}{optim_jl-BackTracking}":
        return r"\multirow[c]{3}{*}{\parbox{1.24cm}{Optim.jl\\Back-\\tracking}}"
    else:
        raise ValueError(rf"Cannot rename index label {label}")


text = sys.stdin.read()
lines = text.splitlines()
rows = [line.split("&") for line in lines]

for row in rows[:]:
    if len(row) == 1:
        print(row[0])
        rows.remove(row)
    else:
        # We should stop removing lines to keep the lines ending the table.
        break

for row in rows:
    if len(row) == 1:
        if row[0].startswith(r"\cline"):
            row[0] = r"\midrule"
        continue
    row[0] = _rename_index(row[0])

widths = [
    max(len(r[i].strip()) for r in rows if len(r) > i)
    for i in range(max((len(r) for r in rows), default=0))
]

for r in rows:
    print(
        " & ".join(r[i].strip().ljust(widths[i]) for i in range(len(r)))
        if len(r) > 1
        else r[0]
    )
