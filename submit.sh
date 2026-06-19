#!/bin/sh

echo "======= Started at  `date` ======="
echo
subabqpy2025-old -p q36 -c 20 -m 120 -t 04:00:00 run_mesh_convergence
echo
echo "======= Finished at `date` ======="
