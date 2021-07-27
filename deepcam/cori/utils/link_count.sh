#!/bin/bash

subdir=(train validation)
samples_per_node=(2360 262) # 20985 in total

node_count=$1
#base_dir="/global/cscratch1/sd/tkurth/data/cam5_data/All-Hist/train"
#base_dir="/global/cscratch1/sd/tkurth/data/cam5_data/All-Hist/all"
#base_dir="/global/cfs/cdirs/mpccc/gsharing/sfarrell/climate-data/All-Hist/train"
base_dir="/global/cscratch1/sd/tkurth/data/cam5_data/All-Hist/all"
data_dir="data/cam5_data/All-Hist_small_split_$node_count"
#data_dir="data/cam5_data/All-Hist-all" 
mkdir -p $data_dir

cd $data_dir
#files=`ls $base_dir`
i=0
while read line
do
    files[ $i ]="$line"
    (( i++ ))
done < <(ls $base_dir)

count=`ls $base_dir | wc -l`
i=0
j=0

f="$base_dir/${files[$j]}"
for s in ${subdir[@]}; do 
	mkdir $s
	cd $s
	samples=${samples_per_node[$i]}
	echo "$node_count:$samples"
	f_count=$((node_count*samples))
	echo $f_count
	k=0
	while (( k < f_count ))
	do
		f="$base_dir/${files[$j]}"
#		echo "$j:$f"
		ln -s $f .
		j=$((j+1))
		k=$((k+1))
	done
	cd ..
	i=$((i+1))
done
