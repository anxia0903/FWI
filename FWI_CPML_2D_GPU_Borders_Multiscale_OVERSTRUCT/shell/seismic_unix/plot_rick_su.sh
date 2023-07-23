#!/bin/sh

# Plot ricker wave using the SU software...

# the file name is called <rick.dat>

# 1. change the Format ascii to binary format using the command "a2b"...
a2b < ../output/rick_amp.dat n1=2 > rick.bin

# 2. Plot the .bin file using the command "xgraph"
xgraph < ../output/rick.bin n=1024

# n=1500 is the length of the ricker wave...
