#!/bin/sh

echo "======= Started at  `date` ======="
echo
subabqpy2025-old -p q128 -c 48 -m 120 -t 10:00:00 run_parameter_study 
echo
echo "======= Finished at `date` ======="
