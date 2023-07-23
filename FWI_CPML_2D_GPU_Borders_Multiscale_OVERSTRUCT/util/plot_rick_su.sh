#!/bin/sh

# Plot ricker wave using the SU software...

# the file name is called <rick.dat>

# 1. change the Format ascii to binary format using the command "a2b"...
a2b < seismogram.dat n1=2 > seismogram.bin
a2b < rick_wave.dat n1=2 > rick.bin
a2b < rick_wave_target.dat n1=2 > rick_target.bin
a2b < rick_wave_filted.dat n1=2 > rick_filted.bin


a2b < rick_wave_amp.dat n1=2 > rick_amp.bin
a2b < rick_wave_target_amp.dat n1=2 > rick_target_amp.bin
a2b < rick_wave_filted_amp.dat n1=2 > rick_filted_amp.bin

# 2. Plot the .bin file using the command "xgraph"
xgraph < seismogram.bin n=2000
xgraph < rick.bin n=2000
xgraph < rick_target.bin n=2000
xgraph < rick_filted.bin n=2000

xgraph < rick_amp.bin n=100
xgraph < rick_target_amp.bin n=100
xgraph < rick_filted_amp.bin n=100

# n=2000 is the length of the ricker wave...
