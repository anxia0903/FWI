#!/bin/bash

#plot the raw model
ximage < ../../output/acc_vp.dat n1=187 d1=15 n2=801 d2=15 legend=1

#plot the inverted model.
for ((i=1;i<=6;i++))
do
    echo "${i}"
    ximage < ../../output/${i}freq_vp.dat n1=187 d1=15 n2=801 d2=15 legend=1
    ximage < ../../output/${i}Gradient_vp.dat n1=801 d1=15 n2=187 d2=15 legend=1
done 
