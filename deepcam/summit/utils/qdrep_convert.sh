#!/bin/bash

CURRENTPATH=$1

module load nsight-systems

FILES=$CURRENTPATH/*.qdrep
OUTPUTEXTENSION=json

for f in $FILES
do
    filename=$(basename -- "$f")
    extension="${filename##*.}"
    filename="${filename%.*}"
    # Export to format
    #echo "Processing $filename.qdrep --> $filename.$OUTPUTEXTENSION ... "
    #nsys export --type=$OUTPUTEXTENSION -o $filename.$OUTPUTEXTENSION $filename.qdrep

    # Save the .sqlite output
    echo "Processing $filename.qdrep --> $filename.txt ... "
    nsys stats $filename.qdrep > $filename.txt
    
done
