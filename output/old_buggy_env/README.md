# Old results (pre-fix environment)

Models, Q-tables, logs and plots in this folder were trained before the
environment and physics fixes: joint limits, link inertias, controller gains,
the reset procedure, the neutral pose, the action set (now 7 leg-group
actions) and the observation layout (now group-state flags) all changed
since. They are kept for reference only. They are not comparable with new
runs, and the saved models and Q-tables will not load correctly into the
current env.

Moved here from:
- `output/*`
- `results/`
- `training_results/`
- `gurdy_qtable_6legs.pkl` (repository root)
