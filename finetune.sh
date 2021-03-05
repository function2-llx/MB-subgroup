#!/usr/bin/bash

pretrained_name=r3d18_KM_200ep

for sample_size in 112 130 150 170 200 224; do
    for sample_slices in 16 20 24; do
        for lr in 2e-5 3e-5 4e-5; do
            for weight_strategy in equal invsqrt; do
                cmd="python run_3d.py \
                    --pretrained_name ${pretrained_name} \
                    --weight_strategy ${weight_strategy} \
                    --sample_size ${sample_size} \
                    --sample_slices ${sample_slices} \
                    --lr ${lr} \
                    --train"
                echo $cmd
                eval $cmd
            done
        done
    done
done