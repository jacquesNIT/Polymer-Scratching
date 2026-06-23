#!/bin/sh

echo "======= Started at  `date` ======="
echo
subabqpy2025-old -p q36 -c 36 -m 120 -t 10:00:00 run_mesh_convergence
echo
echo "======= Finished at `date` ======="
