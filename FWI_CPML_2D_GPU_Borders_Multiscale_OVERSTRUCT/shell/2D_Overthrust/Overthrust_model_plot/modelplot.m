clear all;
clc;
%vmin=1500;
%vmax=5500;
nx=801;
nz=187;
dx=15.0;
dz=15.0;

x=[0:nx-1]*dx;
z=[0:nz-1]*dz;

left=200;
middle=300;
right=550;

 row_cmap = 525;  %����ɫͼ����������
 color_map1=zeros(row_cmap,3);  %����ɫͼ����
for i=1:525
     color_b(i)=1.00;                     %��ɫ-��ɫ 
     color_g(i)=1.00;  
     color_r(i)=1.00;  
end
for i=1:100
     color_r(i)=0.00; 
     color_g(i)=0.00;  
     color_b(i)=0.50+(i-1)*0.005;                   %����ɫ-����ɫ 
end
for i=101:200
    color_r(i)=0.00;
    color_g(i)=0.00+(i-101)*0.01;                   %����ɫ-����ɫ
    color_b(i)=1.00-(i-101)*0.01;   
end
for i=201:300
    color_r(i)=0.00+(i-201)*0.01;    
    color_g(i)=1.00;
    color_b(i)=0.00;                                %����ɫ-��ɫ
end
for i=301:400
    color_r(i)=1.00;                                 %��ɫ-��ɫ
    color_g(i)=1.00-(i-301)*0.01;
    color_b(i)=0.00;
end

for i=401:525
    color_r(i)=1.00;
    color_g(i)=0.00;    
    color_b(i)=0.00+(i-401)*0.008;                  %��ɫ-����ɫ
end


color_map1(:,1) = color_r; 
color_map1(:,2) = color_g;
color_map1(:,3) = color_b;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��׼ȷ��ģ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(111)
fid=fopen('../../../output/acc_vp.dat','rb');
data1=fread(fid,[nz,nx],'float');


pcolor(data1);
colormap(jet);
% colormap(flipud(gray));
%colormap(color_map1);
%caxis([vmin,vmax]);
shading interp;
axis ij;
colorbar; 
size(data1)

a=data1(:,middle);
a1=data1(:,left);
a2=data1(:,right);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%����ʼ��ģ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(113)
fid=fopen('../../../output/ini_vp.dat','rb');
data2=fread(fid,[nz,nx],'float');


pcolor(x,z,data2);
colormap(jet);
% colormap(flipud(gray));
% colormap(color_map1);
%caxis([vmin,vmax]);
shading interp;
axis ij;
colorbar; 

b=data2(:,middle);
b1=data2(:,left);
b2=data2(:,right);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�����ݵ�ģ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(114)
fid=fopen('../../../output/6freq_vp.dat','rb');
data3=fread(fid,[nz,nx],'float');


pcolor(x,z,data3);
colormap(jet);
% colormap(flipud(gray));
% colormap(color_map1);
%caxis([vmin,vmax]);
shading interp;
axis ij;
colorbar; 


c=data3(:,middle);
c1=data3(:,left);
c2=data3(:,right);

x=((1:nz)-1)*dx;
size(c)
figure(15)
plot(a,x,'r',b,x,'b',c,x,'g','linewidth',3);
axis ij;
grid on;

figure(16)
plot(a1,x,'r',b1,x,'b',c1,x,'g','linewidth',3);
axis ij;
grid on;

figure(17)
plot(a2,x,'r',b2,x,'b',c2,x,'g','linewidth',3);
axis ij;
grid on;



