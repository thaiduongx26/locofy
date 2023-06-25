#!/bin/bash

file_list=()
num_epochs=20

for i in {0..4}; do
    filepath="Data/fold_${i}.json"
    file_list+=("$filepath")
done

echo "List of files:"
for filepath in "${file_list[@]}"; do
    echo "$filepath"
done

for filepath in "${file_list[@]}"; do
    filename=$(basename "$filepath" | cut -d. -f1)
    echo "Executing training for $filename"
    python trainer.py --num_epochs $num_epochs --ver_name $filename --do_train --do_eval --do_save
    echo "Finished executing training: $filename"
done

# Traning last time on all data, no evaluation
filepath="Data/all.json"
echo "Executing training for all.json data"
filename="all"
python trainer.py --num_epochs $num_epochs --ver_name $filename --do_train --do_save
echo "Finished executing training: $filename"

echo "Finished executing all training"
