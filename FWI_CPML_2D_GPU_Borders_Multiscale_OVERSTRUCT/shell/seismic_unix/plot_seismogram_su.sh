#!/bin/sh

ximage < ../../output/1source_seismogram_vx_obs.dat n1=801 d1=15 n2=3000 d2=1.5 legend=1
ximage < ../../output/1source_seismogram_vz_obs.dat n1=801 d1=15 n2=3000 d2=1.5 legend=1
ximage < ../../output/1source_seismogram_vx_obs_filted.dat n1=801 d1=15 n2=3000 d2=1.5 legend=1
ximage < ../../output/1source_seismogram_vz_obs_filted.dat n1=801 d1=15 n2=3000 d2=1.5 legend=1





