# This script simulated DNA sequences using `simdna` package
# ZZ, 2020.5.16

SIMDNA_FOREGROUND="../../simdna/scripts/densityMotifSimulation.py"
SIMDNA_BACKGROUND="../../simdna/scripts/emptyBackground.py"

python $SIMDNA_FOREGROUND --prefix MYC_known10 --motifNames MYC_known10 --max-motifs 10 --min-motifs 1 --mean-motifs 5 --seqLength 1000 --numSeqs 10000 --seed 777

python $SIMDNA_FOREGROUND --prefix CTCF_known1 --motifNames CTCF_known1 --max-motifs 1 --min-motifs 1 --mean-motifs 1 --seqLength 1000 --numSeqs 10000 --seed 777

python $SIMDNA_BACKGROUND --prefix empty_bg --seqLength 1000 --numSeqs 10000
