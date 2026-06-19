#!/bin/sh

echo "======= Started at  `date` ======="
echo
subabqpy2025-old -p q36 -c 20 -m 120 -t 02:00:00 run_mass_scale_convergence
echo
echo "======= Finished at `date` ======="
