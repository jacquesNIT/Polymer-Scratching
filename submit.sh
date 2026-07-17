#!/bin/sh

echo "======= Started at  `date` ======="
echo
subabqpy2025-old -p q36 -c 18 -m 80 -t 24:00:00 run_parameter_study 
echo
echo "======= Finished at `date` ======="
