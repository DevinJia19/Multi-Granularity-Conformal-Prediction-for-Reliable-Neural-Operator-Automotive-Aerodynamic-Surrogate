# Ordinary Split-CP 4-Fold Pressure/WSS Notes

`scripts/run_splitcp_4fold.sh` creates four independent ordinary split-CP runs
inside the official 400-case AutoCFD training pool. Each fold trains on 300
cases, calibrates on 100 cases, and evaluates on the official 50-case test
split.

The resulting metrics are averaged across the four independent runs. The
baseline intentionally avoids OOF score merging and final-model refitting.
