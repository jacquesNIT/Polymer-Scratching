#!/bin/sh

echo "======= Started at  `date` ======="
echo
subabqpy2025-old -p q128 -c 18 -m 100 -t 10:00:00 run_parameter_study 
echo
echo "======= Finished at `date` ======="
