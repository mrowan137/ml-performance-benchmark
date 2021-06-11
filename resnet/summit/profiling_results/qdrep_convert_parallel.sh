#!/bin/bash

# USAGE:
# Start interactive session
# jsrun -r 42 -a 1 -c 1 sh ../../../qdrep_convert_parallel.sh . 42
# 12 should be the appropriate number of qdrep files

module load nsight-systems

CURRENTPATH=$1
NRANKS=$2

RANK=$PMIX_RANK
NFILES=`ls -1 *.qdrep 2>/dev/null | wc -l`
FILES=$CURRENTPATH/*.qdrep



i=0
for f in $FILES
do
    if  [[ (($(($i % $NRANKS)) == $RANK)) && (("$i" -lt "$NFILES")) ]]
    then
        echo "Hello I am rank $RANK and I get file $f, i=$i and NFILES=$NFILES"
        FILENAME=$(basename -- "$f")
        extension="${FILENAME##*.}"
        FILENAME="${FILENAME%.*}"
        echo "Processing $FILENAME.qdrep --> $FILENAME.txt ... "
        nsys stats $FILENAME.qdrep > $FILENAME.txt
    fi
    i=$((i + 1))
done
