#!/bin/sh
#SBATCH -J Abaqus-
#SBATCH -o log.out
#SBATCH -p q36
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=20
#SBATCH --mem=120000
#SBATCH --time=00-02:00:00

echo "======= Started at  `date` ======="
echo
subabqpy2025-old -p q36 -c 20 -m 120 -t 02:00:00 run_mass_scale_convergence
echo
echo "======= Finished at `date` ======="
