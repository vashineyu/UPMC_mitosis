#!/bin/bash

for sp in {0..6}
do
echo running spilt $sp

python run.py --gpu_id 4 --split_id $sp --message 'run model training, split_by_file, inception_resnet'
python run.py --gpu_id 4 --split_id $sp --full_random 1 --message 'run model training, random_split'

done
