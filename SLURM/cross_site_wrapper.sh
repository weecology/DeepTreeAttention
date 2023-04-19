#cross_site_wrapper
#!/bin/bash
SITES='["MLBS"] ["BLAN"] ["UKFS"] ["SCBI"] ["BART"] ["HARV"] ["SERC"] ["GRSM"] ["NIWO"] ["RMNP"] ["WREF"] ["SJER"] ["BONA"] ["DEJU"] ["GUAN"] ["TREE"] ["STEI"] ["UNDE"] ["DELA"] ["LENO"] ["CLBJ"] ["TEAK"] ["SOAP"] ["YELL"] ["MOAB"] ["REDB"] ["OSBS"] ["JERC"] ["TALL"] ["DSNY"]'
#SITES='["BART"]'
for SITE in $SITES; do
echo "${SITE}"
sbatch SLURM/experiment.sh $1 ${SITE}
sleep 1 # pause to be kind to the scheduler
done
