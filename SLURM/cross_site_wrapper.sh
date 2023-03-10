#cross_site_wrapper
#!/bin/bash
#SITES='["MLBS"_"BLAN"_"UKFS"] ["SCBI"] ["BART"_"HARV"] ["SERC"_"GRSM"] ["NIWO"_"RMNP"] ["WREF"] ["SJER"] ["BONA"_"DEJU"] ["GUAN"] ["TREE"_"STEI"_"UNDE"] ["DELA"_"LENO"] ["CLBJ"] ["TEAK"_"SOAP"_"YELL"] ["MOAB"_"REDB"] ["OSBS"_"JERC"_"TALL"_"DSNY"]'
SITES='["SCBI"]'
for SITE in $SITES; do
echo "${SITE}"
sbatch SLURM/experiment.py $1 $2 "${SITE}"
sleep 1 # pause to be kind to the scheduler
done