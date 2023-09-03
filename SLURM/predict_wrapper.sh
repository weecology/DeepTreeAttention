#!/bin/bash
SITES="OSBS MLBS BLAN UKFS BART TREE STEI CLBJ TEAK YELL TALL"
#SITES="UNDE OSBS"
for SITE in $SITES; do
echo "${SITE}"
sbatch SLURM/predict.sh ${SITE}
sleep 30 # pause to be kind to the scheduler
done
