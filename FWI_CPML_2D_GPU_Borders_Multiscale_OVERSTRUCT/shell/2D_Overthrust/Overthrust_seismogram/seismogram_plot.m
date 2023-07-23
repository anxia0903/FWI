clear all;
clc;
vmin=-6.0e-10;
vmax=6.0e-10;
fid=fopen('../../../output/5source_seismogram_vz_obs_filted_transfer.dat','rb');
nx=801;
nt=3000;

dx=15.0;
dt=1.0e-3;

x=[0:nx-1]*dx;
t=[0:nt-1]*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��׼ȷ��ģ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(13)

data=fread(fid,[nt,nx],'float');


pcolor(x,t,data);
colormap(gray);
colormap(flipud(gray));
caxis([vmin,vmax]);
shading interp;
axis ij;
colorbar; 









