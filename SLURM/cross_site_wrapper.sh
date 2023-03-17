#cross_site_wrapper
#!/bin/bash
#SITES='["MLBS"_"BLAN"_"UKFS"] ["SCBI"] ["BART"] ["HARV"] ["SERC"_"GRSM"] ["NIWO"_"RMNP"] ["WREF"] ["SJER"] ["BONA"_"DEJU"] ["GUAN"] ["TREE"_"STEI"_"UNDE"] ["DELA"_"LENO"] ["CLBJ"] ["TEAK"_"SOAP"_"YELL"] ["MOAB"_"REDB"] ["OSBS"_"JERC"_"TALL"_"DSNY"]'
SITES='["SERC"]'
for SITE in $SITES; do
echo "${SITE}"
sbatch SLURM/experiment.sh $1 ${SITE}
sleep 1 # pause to be kind to the scheduler
done