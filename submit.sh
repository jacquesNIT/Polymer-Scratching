#!/bin/sh

echo "======= Started at  `date` ======="
echo
HOME/bin/subabqpy_mine -p q36 -c 6 -m 100 -t 4-00:00:00 run_parameter_study
echo
echo "======= Finished at `date` ======="
