#!/usr/bin/env zsh

python plot_grouped_losses.py
# python plot_log_loss.py
python plot_predictions.py 01
python plot_together_grouped_losses.py

python table_statistics.py | python transform-latex-table.py
