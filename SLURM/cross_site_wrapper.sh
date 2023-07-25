#cross_site_wrapper
#!/bin/bash
SITES='OSBS BLAN UKFS SCBI BART HARV SERC GRSM NIWO RMNP WREF SJER BONA DEJU TREE STEI UNDE DELA LENO CLBJ TEAK SOAP YELL REDB MLBS JERC TALL'
for SITE in $SITES; do
echo "${SITE}"
sbatch SLURM/experiment.sh $1 ${SITE}
sleep 1 # pause to be kind to the scheduler
done
