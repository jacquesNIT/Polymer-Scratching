#!/bin/sh

echo "======= Started at  `date` ======="
echo
subabqpy2025-old -p q64 -c 12 -m 100 -t 4-00:00:00 run_parameter_study 
echo
echo "======= Finished at `date` ======="
