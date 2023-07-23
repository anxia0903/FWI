/*================================================================================
MIT License

Copyright (c) 2023 anxia0903

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
===============================================================================*/

/*================================================================================
   This program is a demo to perform full waveform in the time domain.
   The forward method is elastic wave fdtd, and multiscale is used in
   this program to get the lambda and mu parameters which were completed
   in 2013/04.

   Any question contact anxia0903@gmail.com.
===============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define BLOCK_SIZE 16
#define WITH_SHARED_MEMORY 0
#define EPS 0.0001

#define OUTPUT_SNAP 0              
/* OUTPUT_SNAP ==1 
   Output the snap during the forward modeling!
   OUTPUT_SNAP ==0
   Will not output the snap!
*/

#define MULTI_SCALE 1
/* MULTI_SCALE ==1 
   Perform multi scale full waveform inversion in time domain(MFWI)!
   MULTI_SCALE ==0
   Perform single scale full waveform inversion in time domain(SFWI)!
*/
#define PI 3.1415927

#include "head.h"

void minimum_vector(float *vector, int n, float *minimum_value);
void smooth_model(float *vp, int ntp, int ntx, int ntz, int pml);

///===================================================================
///                  Filter in time domain...
///===================================================================

void Hamming(int N,float *win);
void Blackman_harris(int N,float *win);

void hdn(float *hd,int N,float wc);
void hh(float *win, float *hd, float *h, int N);
void con(float *x, int m, float *h, int n, float *out);
void filter_of_ricker(
     float fpp, float fs, float *rick, 
     int itmax, float *rick_filted, float dt);

void filter_of_seismogram(
     float fpp, float fs, float **seismogram, 
     int itmax, int ntr, float **seismogram_filted, float dt);

///===================================================================
///                  Filter in frequency domain...
///===================================================================

void fre_filter(float *H, int NFFT, float w1, float w2, float df);
void fre_filter_P(float *P, int NFFT, float f1, float f2, float df);
void fft(float *xreal,float *ximag,int n,int sign);

void filter_of_ricker_frequency(
     float fpp, float fs, float *rick, 
     int itmax, float *rick_filted, float dt);

void filter_of_seismogram_frequency(
     float fpp, float fs, float **seismogram, 
     int itmax, int ntr, float **seismogram_filted, float dt);


///===================================================================
///                  Wiener Filter ...
///===================================================================
void get_wiener_filter(
     float *rick_fft_real, float *rick_fft_imag, 
     float *rick_target_fft_real, float *rick_target_fft_imag,
     float *filter_wiener_fft_real,
     float *filter_wiener_fft_imag, int NFFT);

void filter_of_ricker_wiener(
     float *rick_fft_real, float *rick_fft_imag,
     float *filter_wiener_fft_real,
     float *filter_wiener_fft_imag,
     int NFFT, float *rick, int itmax);

void filter_of_seismogram_wiener(
     float **seismogram, int itmax, int ntr, float **seismogram_filted,
     float *filter_wiener_fft_real, float *filter_wiener_fft_imag, int NFFT);


int main()
{

/*=========================================================
  Parameters of the time of the system...
  =========================================================*/
  
  time_t begin_time;

  clock_t start;
  clock_t end;

//  float runtime=0.0f;
//  cudaEvent_t start2,stop;
    
/*=========================================================
  Parameters of Cartesian coordinate...
  ========================================================*/  
  
  int nz=187;
  int nx=801;
  int pml=20;
  
  int ntz=nz+2*pml;
  int ntx=nx+2*pml;
  int ntp=ntz*ntx;

  int ip,iz,ix;
  
  float dz=15.0f;
  float dx=15.0f;
  
/*=========================================================
  Parameters of ricker wave...
  ========================================================*/
  
  int   it;
  int   itmax=3000;
  float *rick,*rick_old;
  float f0=20.0f;
  float t0=0.20f;
  float dt=1.0e-3f;
  
/*=========================================================
  Parameters of model...
  ========================================================*/
  
  float *vp,*vs,*rho;
  float *lambda,*mu,*lambda_plus_two_mu;
  float *vp_max;
  float *vp_n,*vs_n;
  
/*=========================================================
  Parameters of absorbing layers...
  ========================================================*/
  
//  float thickness_of_pml=pml*dx;
//  float Rc=1.0e-3;
//  float Vpmax=3300.0;
//  float d0;
  
  float *d_x,*d_x_half,*d_z,*d_z_half;
  float *a_x,*a_x_half,*a_z,*a_z_half;
  float *b_x,*b_x_half,*b_z,*b_z_half;
  float *k_x,*k_x_half,*k_z,*k_z_half;
  
/*=========================================================
  Parameters of Sources and Receivers...
  ========================================================*/
  int is;
  int ns=30;
  int *s_iz;
  int *s_ix;

//  int source_iz=pml+8;
//  int source_ix=(ntx+1)/2;

/*=========================================================
  Parameters of the coefficients of the space...
  ========================================================*/
  
  float c[2]={9.0f/8.0f,-1.0f/24.0f};


/*=========================================================
  Parameters of Seismograms and Borders...
  ========================================================*/

  float Misfit_old=0.0f;

  float **seismogram_vx_obs_all_sources;
  float **seismogram_vz_obs_all_sources;

  float **seismogram_vx_obs_all_sources_filted;
  float **seismogram_vz_obs_all_sources_filted;

  float *seismogram_vx_obs, *seismogram_vz_obs;
  float *seismogram_vx_obs_T, *seismogram_vz_obs_T;
  float *seismogram_vx_syn, *seismogram_vz_syn;
  float *seismogram_vx_rms, *seismogram_vz_rms;

  float **seis_vx_obs,**seis_vz_obs;
  float **seis_vx_obs_filted,**seis_vz_obs_filted;

  float *vx_borders_up,   *vx_borders_bottom;
  float *vx_borders_left, *vx_borders_right;

  float *vz_borders_up,   *vz_borders_bottom;
  float *vz_borders_left, *vz_borders_right;

/*=========================================================
  Image / gradient ...
 *========================================================*/

  float *image_lambda,*image_mu;
  float *image_sources,*image_receivers;

  float *Gradient_vp_pre;
  float *Gradient_vs_pre;
  float *dn_vp_pre;
  float *dn_vs_pre;
  int    np=nx*nz;

/*=========================================================
  Parameters of the time filter...
  ========================================================*/
  
  int   ifreq,Nfreq;
  float freq_s[10];
  float freq_e[10];

/*=========================================================
  Parameters of the wiener filter...
  ========================================================*/

  int   K,NFFT;
  
  float *rick_fft_real,*rick_fft_imag;
  float *rick_target_fft_real,*rick_target_fft_imag;
  float *filter_wiener_fft_real,*filter_wiener_fft_imag;

  float f0_target[10];
  float t0_target[10];

  float df;

/*=========================================================
  File name....
 *========================================================*/

  FILE *fp;
  char filename[230];

/*=========================================================
  Flags ....
 *========================================================*/

  int inv_flag;
  int iter,N_iter;
  
//#######################################################################
// NOW THE PROGRAM BEGIN
//#######################################################################

  time(&begin_time);
  printf("Today's data and time: %s",ctime(&begin_time));
  
/*=========================================================
  Allocate the memory of parameters of ricker wave...
  ========================================================*/
  
  rick=(float*)malloc(sizeof(float)*itmax);
  rick_old=(float*)malloc(sizeof(float)*itmax);
  
/*=========================================================
  Allocate the memory of parameters of model...
  ========================================================*/
  
  // allocate the memory of model parameters...
  
  vp_max              = (float*)malloc(sizeof(float)*1);
  vp                  = (float*)malloc(sizeof(float)*ntp);
  vs                  = (float*)malloc(sizeof(float)*ntp);
  rho                 = (float*)malloc(sizeof(float)*ntp);
    
  lambda              = (float*)malloc(sizeof(float)*ntp);
  mu                  = (float*)malloc(sizeof(float)*ntp);
  lambda_plus_two_mu  = (float*)malloc(sizeof(float)*ntp);
  
  vp_n                = (float*)malloc(sizeof(float)*ntp);
  vs_n                = (float*)malloc(sizeof(float)*ntp);
  
/*=========================================================
  Allocate the memory of parameters of absorbing layer...
  ========================================================*/
  
  d_x      = (float*)malloc(ntx*sizeof(float));
  d_x_half = (float*)malloc(ntx*sizeof(float));    
  d_z      = (float*)malloc(ntz*sizeof(float));
  d_z_half = (float*)malloc(ntz*sizeof(float));
  
  
  a_x      = (float*)malloc(ntx*sizeof(float));
  a_x_half = (float*)malloc(ntx*sizeof(float));    
  a_z      = (float*)malloc(ntz*sizeof(float));
  a_z_half = (float*)malloc(ntz*sizeof(float));
  
  
  b_x      = (float*)malloc(ntx*sizeof(float));
  b_x_half = (float*)malloc(ntx*sizeof(float));
  b_z      = (float*)malloc(ntz*sizeof(float));
  b_z_half = (float*)malloc(ntz*sizeof(float));
  
  
  k_x      = (float*)malloc(ntx*sizeof(float));
  k_x_half = (float*)malloc(ntx*sizeof(float));
  k_z      = (float*)malloc(ntz*sizeof(float));
  k_z_half = (float*)malloc(ntz*sizeof(float));
  
  
/*=========================================================
  Allocate the memory of Seismograms...
  ========================================================*/
    
  seismogram_vx_obs=(float*)malloc(sizeof(float)*itmax*nx);
  seismogram_vz_obs=(float*)malloc(sizeof(float)*itmax*nx);

  seismogram_vx_obs_T=(float*)malloc(sizeof(float)*itmax*nx);
  seismogram_vz_obs_T=(float*)malloc(sizeof(float)*itmax*nx);

  seismogram_vx_obs_all_sources=(float**)malloc(sizeof(float*)*ns);
  seismogram_vz_obs_all_sources=(float**)malloc(sizeof(float*)*ns);

  seismogram_vx_obs_all_sources_filted=(float**)malloc(sizeof(float*)*ns);
  seismogram_vz_obs_all_sources_filted=(float**)malloc(sizeof(float*)*ns);

  for(is=0;is<ns;is++)
  {
      seismogram_vx_obs_all_sources[is]=(float*)malloc(sizeof(float)*nx*itmax);
      seismogram_vz_obs_all_sources[is]=(float*)malloc(sizeof(float)*nx*itmax);
  
      seismogram_vx_obs_all_sources_filted[is]=(float*)malloc(sizeof(float)*nx*itmax);
      seismogram_vz_obs_all_sources_filted[is]=(float*)malloc(sizeof(float)*nx*itmax);
  }

  seis_vx_obs=(float**)malloc(sizeof(float*)*itmax);
  seis_vz_obs=(float**)malloc(sizeof(float*)*itmax);

  seis_vx_obs_filted=(float**)malloc(sizeof(float*)*itmax);
  seis_vz_obs_filted=(float**)malloc(sizeof(float*)*itmax);

  for(it=0;it<itmax;it++)
  {
      seis_vx_obs[it]=(float*)malloc(sizeof(float)*nx);
      seis_vz_obs[it]=(float*)malloc(sizeof(float)*nx);
  
      seis_vx_obs_filted[it]=(float*)malloc(sizeof(float)*nx);
      seis_vz_obs_filted[it]=(float*)malloc(sizeof(float)*nx);
  }

  seismogram_vx_syn=(float*)malloc(sizeof(float)*itmax*nx);
  seismogram_vz_syn=(float*)malloc(sizeof(float)*itmax*nx);

  seismogram_vx_rms=(float*)malloc(sizeof(float)*itmax*nx);
  seismogram_vz_rms=(float*)malloc(sizeof(float)*itmax*nx);


  vx_borders_up    =(float*)malloc(sizeof(float)*itmax*(nx+1));
  vx_borders_bottom=(float*)malloc(sizeof(float)*itmax*(nx+1));
  vx_borders_left  =(float*)malloc(sizeof(float)*itmax*(nz-2));
  vx_borders_right =(float*)malloc(sizeof(float)*itmax*(nz-2));

  vz_borders_up    =(float*)malloc(sizeof(float)*itmax*(nx));
  vz_borders_bottom=(float*)malloc(sizeof(float)*itmax*(nx));
  vz_borders_left  =(float*)malloc(sizeof(float)*itmax*(nz-1));
  vz_borders_right =(float*)malloc(sizeof(float)*itmax*(nz-1));

/*=========================================================
  Allocate the memory of image / gradient...
  ========================================================*/

  image_lambda=(float*)malloc(sizeof(float)*ntp);
  image_mu=(float*)malloc(sizeof(float)*ntp);

  image_sources=(float*)malloc(sizeof(float)*ntp);
  image_receivers=(float*)malloc(sizeof(float)*ntp);

  dn_vp_pre      =(float*)malloc(sizeof(float)*np);
  dn_vs_pre      =(float*)malloc(sizeof(float)*np);

  Gradient_vp_pre=(float*)malloc(sizeof(float)*np);
  Gradient_vs_pre=(float*)malloc(sizeof(float)*np);
  
/*=========================================================
  Calculate the sources' poisition...
  ========================================================*/

  s_iz=(int*)malloc(sizeof(int)*ns);
  s_ix=(int*)malloc(sizeof(int)*ns);

  for(is=0;is<ns;is++)
  {
     s_iz[is]=pml+3;
     s_ix[is]=pml+40+(is+1)*24;
  }

/*=========================================================
  Calculate the ricker wave...
  ========================================================*/
  
  ricker_wave(rick,itmax,f0,t0,dt);
  printf("Ricker wave is done\n");
  
  for(it=0;it<itmax;it++)
  {
      rick_old[it]=rick[it];
  }

/*=========================================================
  Calculate the ture model.../Or read in the true model
  ========================================================*/

  inv_flag=0;
  get_acc_model(vp,vs,rho,ntp,ntx,ntz);

  fp=fopen("./output/acc_vp.dat","wb");
  for(ix=pml;ix<=ntx-pml-1;ix++)
  {
      for(iz=pml;iz<=ntz-pml-1;iz++)
      {
          fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);  
      }
  }
  fclose(fp);

  fp=fopen("./output/acc_vs.dat","wb");
  for(ix=pml;ix<=ntx-pml-1;ix++)
  {
      for(iz=pml;iz<=ntz-pml-1;iz++)
      {
          fwrite(&vs[iz*ntx+ix],sizeof(float),1,fp);   
      }
  }
  fclose(fp);

  fp=fopen("./output/acc_rho.dat","wb");
  for(ix=pml;ix<=ntx-pml-1;ix++)
  {
      for(iz=pml;iz<=ntz-pml-1;iz++)
      {
          fwrite(&rho[iz*ntx+ix],sizeof(float),1,fp);   
      }
  }
  fclose(fp);

  maximum_vector(vp,ntp,vp_max);
  get_lame_constants(lambda,mu,lambda_plus_two_mu,vp,vs,rho,
                     ntp);
  
  printf("The true model is done\n");

/*=========================================================
  Calculate the parameters of absorbing layers...
  ========================================================*/
  
  get_absorbing_parameters(
                           d_x,d_x_half,d_z,d_z_half,
                           a_x,a_x_half,a_z,a_z_half,
                           b_x,b_x_half,b_z,b_z_half,
                           k_x,k_x_half,k_z,k_z_half,
                           ntz,ntx,nz,nx,pml,dx,f0,t0,
                           dt,vp_max
                          );
  
  printf("ABC parameters are done\n");
 
  start=clock();

  /*=======================================================
  Calculate the Observed seismograms...
  ========================================================*/

  for(is=0;is<ns;is++)
  {
      fdtd_cpml_2d_GPU(ntx,ntz,ntp,nx,nz,pml,dx,dz,
                       rick,itmax,dt,
                       is,s_ix[is],s_iz[is],
                       rho,lambda,mu,lambda_plus_two_mu,
                       k_x,k_x_half,k_z,k_z_half,
                       a_x,a_x_half,a_z,a_z_half,
                       b_x,b_x_half,b_z,b_z_half,c,
                       inv_flag,
                       seismogram_vx_syn,seismogram_vz_syn,
                       vx_borders_up,vx_borders_bottom,
                       vx_borders_left,vx_borders_right,
                       vz_borders_up,vz_borders_bottom,
                       vz_borders_left,vz_borders_right);
  }

  printf("Observed seismogram has been done\n");

  for(is=0;is<ns;is++)
  {
      // READ IN OBSERVED SEISMOGRAMS...
      sprintf(filename,"./output/%dsource_seismogram_vx_obs.dat",is+1);
      fp=fopen(filename,"rb");
      fread(&seismogram_vx_obs_all_sources[is][0],sizeof(float),nx*itmax,fp);
      fclose(fp);

      sprintf(filename,"./output/%dsource_seismogram_vz_obs.dat",is+1);
      fp=fopen(filename,"rb");
      fread(&seismogram_vz_obs_all_sources[is][0],sizeof(float),nx*itmax,fp);
      fclose(fp);

      for(it=0;it<itmax;it++)
      {
          for(ix=0;ix<nx;ix++)
          {
              ip=it*nx+ix;
              seis_vx_obs[it][ix]=seismogram_vx_obs_all_sources[is][ip];
              seis_vz_obs[it][ix]=seismogram_vz_obs_all_sources[is][ip];
          }
      }

      for(ix=0;ix<nx;ix++)
      {
          for(it=0;it<itmax;it++)
          {
              ip=ix*itmax+it;
              seismogram_vx_obs_T[ip]=seis_vx_obs[it][ix];
              seismogram_vz_obs_T[ip]=seis_vz_obs[it][ix];
          }
      }
      sprintf(filename,"./output/%dsource_seismogram_vx_obs_transfer.dat",is+1);
      fp=fopen(filename,"wb");
      fwrite(&seismogram_vx_obs_T[0],sizeof(float),nx*itmax,fp);
      fclose(fp);

      sprintf(filename,"./output/%dsource_seismogram_vz_obs_transfer.dat",is+1);
      fp=fopen(filename,"wb");
      fwrite(&seismogram_vz_obs_T[0],sizeof(float),nx*itmax,fp);
      fclose(fp);
   }

/*=======================================================
  Get the group frequencies 
========================================================*/

/*   For time and frequency filter....
  if(MULTI_SCALE==1)
  {

     for(ifreq=0;ifreq<10;ifreq++)
     {
         freq_s[ifreq]=0.0;
         freq_e[ifreq]=0.0;
     }
     freq_e[0]=10;
     freq_e[1]=15;
     freq_e[2]=30;
     freq_e[3]=42;
     freq_e[4]=54;
  }
*/
  if(MULTI_SCALE==1)
  {
     for(ifreq=0;ifreq<10;ifreq++)
     {
         f0_target[ifreq]=0.0;
         t0_target[ifreq]=t0;
     }
     f0_target[0]=5.0;
     f0_target[1]=8.0;
     f0_target[2]=10.0;    
     f0_target[3]=13.0;
     f0_target[4]=16.0;
     f0_target[5]=20.0;
  }

  if(MULTI_SCALE==1)
  {
     Nfreq=6;
     N_iter=10;
  }
  else
  {
     Nfreq=1;
     N_iter=60;
  }

  K=ceil(log(1.0*itmax)/log(2.0));
  NFFT=pow(2.0,K);
  df=1.0/(NFFT*dt);

  rick_fft_real=(float*)malloc(sizeof(float)*NFFT);
  rick_fft_imag=(float*)malloc(sizeof(float)*NFFT);

  rick_target_fft_real=(float*)malloc(sizeof(float)*NFFT);
  rick_target_fft_imag=(float*)malloc(sizeof(float)*NFFT);

  filter_wiener_fft_real=(float*)malloc(sizeof(float)*NFFT);
  filter_wiener_fft_imag=(float*)malloc(sizeof(float)*NFFT);

  for(it=0;it<NFFT;it++)
  {
      rick_fft_real[it]=0.0;
      rick_fft_imag[it]=0.0;
  }  
   
  for(it=0;it<itmax;it++)
  {
      rick_fft_real[it]=rick_old[it];
  }

  fft(rick_fft_real, rick_fft_imag, NFFT, 1);

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!	        ITERATION OF FWI IN TIME DOMAIN BEGINS...                      
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

  for(ifreq=0;ifreq<Nfreq;ifreq++)  // Loop of the frequency...
  {

      if(MULTI_SCALE==1)
      {
         printf("****************************************************\n");
         printf(" FRENQUENCY == %d\n",ifreq+1);

         for(it=0;it<itmax;it++)
         {
             rick[it]=0.0;
         }

         /*============================================================
           filte the ricker wave
         *============================================================*/

         /* filter_of_ricker_frequency(freq_s[ifreq],freq_e[ifreq],rick_old,itmax,rick,dt); */

         ricker_wave(rick,itmax,f0_target[ifreq],t0_target[ifreq],dt);
         printf("%12.8f\n",f0_target[ifreq]);
         
         fp=fopen("rick_wave_target.dat","w");    
         for(it=0;it<itmax;it++)
         {
             fprintf(fp,"%18f%18f\n",it*dt,rick[it]);
         }     
         fclose(fp);

         for(it=0;it<NFFT;it++)
         {
             rick_target_fft_real[it]=0.0;
             rick_target_fft_imag[it]=0.0;
         }  
   
         for(it=0;it<itmax;it++)
         {
             rick_target_fft_real[it]=rick[it];
         }
         fft(rick_target_fft_real, rick_target_fft_imag, NFFT, 1);
      
         
         get_wiener_filter(rick_fft_real,rick_fft_imag,
                           rick_target_fft_real,rick_target_fft_imag,
                           filter_wiener_fft_real,filter_wiener_fft_imag,NFFT);


         filter_of_ricker_wiener(rick_fft_real,rick_fft_imag,
                                 filter_wiener_fft_real,
                                 filter_wiener_fft_imag,NFFT,rick,itmax);

         fp=fopen("rick_wave_filted.dat","w");    
         for(it=0;it<itmax;it++)
         {
             fprintf(fp,"%18f%18f\n",it*dt,rick[it]);
         }     
         fclose(fp);

         printf(" Ricker wave has been filted...\n");

         for(is=0;is<ns;is++)
         {
             for(it=0;it<itmax;it++)
             {
                 for(ix=0;ix<nx;ix++)
                 {
                     ip=it*nx+ix;
                     seis_vx_obs[it][ix]=seismogram_vx_obs_all_sources[is][ip];
                     seis_vz_obs[it][ix]=seismogram_vz_obs_all_sources[is][ip];
                 }
             }

/*  
             Filter of the frequency...       
             filter_of_seismogram_frequency(freq_s[ifreq],freq_e[ifreq],seis_vx_obs,itmax,nx,seis_vx_obs_filted,dt);
             filter_of_seismogram_frequency(freq_s[ifreq],freq_e[ifreq],seis_vz_obs,itmax,nx,seis_vz_obs_filted,dt);
*/

             filter_of_seismogram_wiener(seis_vx_obs,itmax,nx,seis_vx_obs_filted,
                                         filter_wiener_fft_real,filter_wiener_fft_imag,NFFT);

             filter_of_seismogram_wiener(seis_vz_obs,itmax,nx,seis_vz_obs_filted,
                                         filter_wiener_fft_real,filter_wiener_fft_imag,NFFT);


             for(it=0;it<itmax;it++)
             {
                 for(ix=0;ix<nx;ix++)
                 {
                     ip=it*nx+ix;
                     seismogram_vx_obs_all_sources_filted[is][ip]=seis_vx_obs_filted[it][ix];
                     seismogram_vz_obs_all_sources_filted[is][ip]=seis_vz_obs_filted[it][ix];
                 }
             }

             sprintf(filename,"./output/%dsource_seismogram_vx_obs_filted.dat",is+1);
             fp=fopen(filename,"wb");
             fwrite(&seismogram_vx_obs_all_sources_filted[is][0],sizeof(float),nx*itmax,fp);
             fclose(fp);

             sprintf(filename,"./output/%dsource_seismogram_vz_obs_filted.dat",is+1);
             fp=fopen(filename,"wb");
             fwrite(&seismogram_vz_obs_all_sources_filted[is][0],sizeof(float),nx*itmax,fp);
             fclose(fp);

             for(ix=0;ix<nx;ix++)
             {
                 for(it=0;it<itmax;it++)
                 {
                     ip=ix*itmax+it;
                     seismogram_vx_obs_T[ip]=seis_vx_obs_filted[it][ix];
                     seismogram_vz_obs_T[ip]=seis_vz_obs_filted[it][ix];
                 }
             }
             sprintf(filename,"./output/%dsource_seismogram_vx_obs_filted_transfer.dat",is+1);
             fp=fopen(filename,"wb");
             fwrite(&seismogram_vx_obs_T[0],sizeof(float),nx*itmax,fp);
             fclose(fp);

             sprintf(filename,"./output/%dsource_seismogram_vz_obs_filted_transfer.dat",is+1);
             fp=fopen(filename,"wb");
             fwrite(&seismogram_vz_obs_T[0],sizeof(float),nx*itmax,fp);
             fclose(fp);

         }

         printf(" Seismogram has been filted...\n");
      }   

      for(iter=0;iter<N_iter;iter++)
      {
  
          printf("====================\n");
          printf("ITERATION == %d\n",iter+1);

          /*=======================================================
           Calculate the Synthetic seismograms, and save the borders 
           into the HOST MEMORY.... RMS of Vx and Vz are computed...
          ========================================================*/
          if(ifreq==0)
          {
             if(iter==0)
             {
                get_ini_model(vp,vs,rho,vp_n,vs_n,ntp,ntx,ntz);

                fp=fopen("./output/ini_vp.dat","wb");
                for(ix=pml;ix<=ntx-pml-1;ix++)
                {
                    for(iz=pml;iz<=ntz-pml-1;iz++)
                    {
                        fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);  
                    }
                }
                fclose(fp);

                for(ip=0;ip<ntp;ip++)
                {
                    vp_n[ip]=vp[ip];
                    vs_n[ip]=vs[ip];
                }
         
                get_lame_constants(lambda,mu,lambda_plus_two_mu,vp,vs,rho,
                                   ntp);
             }
           }

          if(ifreq>=1)
          {
             if(iter==0)
             {
                //get_ini_model_freq(vp,vs,rho,vp_n,vs_n,ntp,ntx,ntz);

                for(ip=0;ip<ntp;ip++)
                {
                    vp_n[ip]=vp[ip];
                    vs_n[ip]=vs[ip];
                }
         
                get_lame_constants(lambda,mu,lambda_plus_two_mu,vp,vs,rho,
                                   ntp);
             }
          }



        for(ip=0;ip<ntp;ip++)
        {
            image_lambda[ip]=0.0f;
            image_mu[ip]=0.0f;
        }

        Misfit_old=0.0f;

        /*=======================================================
          Back-propagate the RMS wavefields and Construct 
          the forward wavefield..Meanwhile the gradients 
          of lambda and mu are computed... 
        ========================================================*/

        inv_flag=1;
        for(is=0;is<ns;is++)
        {

            fdtd_cpml_2d_GPU(ntx,ntz,ntp,nx,nz,pml,dx,dz,
                           rick,itmax,dt,
                           is,s_ix[is],s_iz[is],
                           rho,lambda,mu,lambda_plus_two_mu,
                           k_x,k_x_half,k_z,k_z_half,
                           a_x,a_x_half,a_z,a_z_half,
                           b_x,b_x_half,b_z,b_z_half,c,
                           inv_flag,
                           seismogram_vx_syn,seismogram_vz_syn,
                           vx_borders_up,vx_borders_bottom,
                           vx_borders_left,vx_borders_right,
                           vz_borders_up,vz_borders_bottom,
                           vz_borders_left,vz_borders_right
                           );


       /*=======================================================
    
       ========================================================*/
/*
            maximum_vector(seismogram_vx_obs,itmax*nx,vp_max);

            for(ip=0;ip<itmax*nx;ip++)
            {
                    seismogram_vx_obs[ip]=seismogram_vx_obs[ip]/(*vp_max);
            }

            maximum_vector(seismogram_vz_obs,itmax*nx,vp_max);
            for(ip=0;ip<itmax*nx;ip++)
            {
                    seismogram_vz_obs[ip]=seismogram_vz_obs[ip]/(*vp_max);
                
            }

            maximum_vector(seismogram_vx_syn,itmax*nx,vp_max);
            for(ip=0;ip<itmax*nx;ip++)
            {
                    seismogram_vx_syn[ip]=seismogram_vx_syn[ip]/(*vp_max);
                
            }

            maximum_vector(seismogram_vz_syn,itmax*nx,vp_max);
            for(ip=0;ip<itmax*nx;ip++)
            {
                    seismogram_vz_syn[ip]=seismogram_vz_syn[ip]/(*vp_max);
                
            }
*/

            if(MULTI_SCALE==1)
            {
               for(ip=0;ip<nx*itmax;ip++)
               {
                   seismogram_vx_rms[ip]=seismogram_vx_obs_all_sources_filted[is][ip]-seismogram_vx_syn[ip];
                   seismogram_vz_rms[ip]=seismogram_vz_obs_all_sources_filted[is][ip]-seismogram_vz_syn[ip];
               }
            }

            else
            {          
               for(ip=0;ip<nx*itmax;ip++)
               {
                   seismogram_vx_rms[ip]=seismogram_vx_obs_all_sources[is][ip]-seismogram_vx_syn[ip];
                   seismogram_vz_rms[ip]=seismogram_vz_obs_all_sources[is][ip]-seismogram_vz_syn[ip];
               }
            }

            for(ip=0;ip<nx*itmax;ip++)
            {
                Misfit_old=Misfit_old+
                (seismogram_vx_rms[ip]*seismogram_vx_rms[ip])+
                (seismogram_vz_rms[ip]*seismogram_vz_rms[ip]);
            }

            /*Only used to test...*/
            sprintf(filename,"./output/%dsource_seismogram_vx_syn.dat",is+1);
            fp=fopen(filename,"wb");
            fwrite(&seismogram_vx_syn[0],sizeof(float),nx*itmax,fp);
            fclose(fp);

            sprintf(filename,"./output/%dsource_seismogram_vz_syn.dat",is+1);
            fp=fopen(filename,"wb");
            fwrite(&seismogram_vz_syn[0],sizeof(float),nx*itmax,fp);
            fclose(fp);

            if(iter==0)
            {
               sprintf(filename,"./output/%dsource_seismogram_vx_rms_initial.dat",is+1);
               fp=fopen(filename,"wb");
               fwrite(&seismogram_vx_rms[0],sizeof(float),nx*itmax,fp);
               fclose(fp);

               sprintf(filename,"./output/%dsource_seismogram_vz_rms_initial.dat",is+1);
               fp=fopen(filename,"wb");
               fwrite(&seismogram_vz_rms[0],sizeof(float),nx*itmax,fp);
               fclose(fp);
            }


        /*=======================================================
        Calculate the IMAGE/GRADIENT OF RTM/FWI...
        ========================================================*/

            fdtd_2d_GPU_backward(ntx,ntz,ntp,nx,nz,pml,dx,dz,
                       rick,itmax,dt,
                       is,s_ix[is],s_iz[is],
                       rho,lambda,mu,lambda_plus_two_mu,
                       k_x,k_x_half,k_z,k_z_half,
                       a_x,a_x_half,a_z,a_z_half,
                       b_x,b_x_half,b_z,b_z_half,
                       seismogram_vx_rms,seismogram_vz_rms,
                       vx_borders_up,vx_borders_bottom,
                       vx_borders_left,vx_borders_right,
                       vz_borders_up,vz_borders_bottom,
                       vz_borders_left,vz_borders_right,
                       image_lambda,image_mu,
                       image_sources,image_receivers
                       );

        }

        Misfit_old=Misfit_old*1.0e+10;
/*
      for(ix=pml;ix<=ntx-pml-1;ix++)
      {
          for(iz=pml;iz<=ntz-pml-1;iz++)
          {
              ip=ix+iz*ntx;
              image_lambda[ip]=image_lambda[ip]/
              (image_receivers[ip]*image_receivers[ip]+100);
          }
      }

*/
        sprintf(filename,"./output/%dGradient_lambda.dat",iter+1);
        fp=fopen(filename,"wb");
        fwrite(&image_lambda[0],sizeof(float),ntp,fp);
        fclose(fp);

        sprintf(filename,"./output/%dGradient_sources.dat",iter+1);
        fp=fopen(filename,"wb");
        fwrite(&image_sources[0],sizeof(float),ntp,fp);
        fclose(fp);

        sprintf(filename,"./output/%dGradient_receivers.dat",iter+1);
        fp=fopen(filename,"wb");
        fwrite(&image_receivers[0],sizeof(float),ntp,fp);
        fclose(fp);
      
        /*=========================================================
          Applied the conjugate gradient method in FWI
        ==========================================================*/

        inv_flag=3;
        conjugate_gradient(image_lambda,image_mu,
                     vp,vs,rho,
                     lambda,mu,lambda_plus_two_mu,
                     vp_n,vs_n,
                     ntp,ntz,ntx,nz,nx,pml,dz,dx,
                     rick,itmax,dt,
                     ns,s_ix,s_iz,
                     k_x,k_x_half,k_z,k_z_half,
                     a_x,a_x_half,a_z,a_z_half,
                     b_x,b_x_half,b_z,b_z_half,
                     c,
                     seismogram_vx_syn,seismogram_vz_syn,
                     vx_borders_up,vx_borders_bottom,
                     vx_borders_left,vx_borders_right,
                     vz_borders_up,vz_borders_bottom,
                     vz_borders_left,vz_borders_right,
                     iter,inv_flag,Misfit_old,
                     Gradient_vp_pre, Gradient_vs_pre,
                     dn_vp_pre, dn_vs_pre);

        for(ip=0;ip<ntp;ip++)
        {
            vp_n[ip]=vp[ip];
            /* vs_n[ip]=vs[ip]; */
        }
    }

        /*==========================================================
          Output the updated model such as vp,vs,...
        ===========================================================*/

    sprintf(filename,"./output/%dfreq_vp.dat",ifreq+1);
    fp=fopen(filename,"wb");
    for(ix=pml;ix<=ntx-pml-1;ix++)
    {
        for(iz=pml;iz<=ntz-pml-1;iz++)
        {
            fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);
        }
    }
    fclose(fp);

  }

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!	        ITERATION OF FWI IN TIME DOMAIN ENDS...                        
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

  free(rick);
  
  free(a_x);
  free(a_x_half);
  free(a_z);
  free(a_z_half);
  
  free(b_x);
  free(b_x_half);
  free(b_z);
  free(b_z_half);
  
  free(d_x);
  free(d_x_half);
  free(d_z);
  free(d_z_half);
  
  free(k_x);
  free(k_x_half);
  free(k_z);
  free(k_z_half);

  free(s_iz);
  free(s_ix);

  //free the memory of P velocity
  free(vp);
  free(vp_n);
  free(vs_n);
  //free the memory of S velocity
  free(vs); 
  //free the memory of Density
  free(rho); 
  //free the memory of lambda
  free(lambda);
  //free the memory of Mu
  free(mu);
  //free the memory of lamda+2Mu
  free(lambda_plus_two_mu);


  free(seismogram_vx_obs);
  free(seismogram_vz_obs);
  free(seismogram_vx_syn);
  free(seismogram_vz_syn); 
  free(seismogram_vx_rms);
  free(seismogram_vz_rms);


  free(vx_borders_up);
  free(vx_borders_bottom);
  free(vx_borders_left);
  free(vx_borders_right);

  free(vz_borders_up);
  free(vz_borders_bottom);
  free(vz_borders_left);
  free(vz_borders_right);

  free(image_lambda);
  free(image_mu);
  free(image_sources);
  free(image_receivers);

  free(dn_vp_pre);
  free(dn_vs_pre);
  free(Gradient_vp_pre);
  free(Gradient_vs_pre);


  end=clock();
  printf("The cost of the run time is %f seconds\n",
         (double)(end-start)/CLOCKS_PER_SEC);
 
}


/*==========================================================
  This subroutine is used for calculating the parameters of 
  absorbing layers
===========================================================*/

  void get_absorbing_parameters(
                      float *d_x, float *d_x_half, 
                      float *d_z, float *d_z_half,
                      float *a_x, float *a_x_half,
                      float *a_z, float *a_z_half,
                      float *b_x, float *b_x_half,
                      float *b_z, float *b_z_half,
                      float *k_x, float *k_x_half,
                      float *k_z, float *k_z_half,
                      int ntz, int ntx, int nz, int nx,
                      int pml, float dx, float f0, float t0, float dt, float *vp_max)
                      
  {
    int   N=2;
    int   iz,ix;
    
    float thickness_of_pml;
    float Rc=1.0e-3f;
       
    float d0;
    float pi=3.1415927f;
    float alpha_max=pi*15;

    float Vpmax;


    float *alpha_x,*alpha_x_half;
    float *alpha_z,*alpha_z_half;
    
    float x_start,x_end,delta_x;
    float z_start,z_end,delta_z;
    float x_current,z_current;
    
    Vpmax=*vp_max;
    Vpmax=5000.0f;

    if(dx>=5.0f&&dx<=10.0f) thickness_of_pml=pml*3;
    if(dx>10.0f) thickness_of_pml=pml*10.0f;
      
    d0=-(N+1)*Vpmax*log(Rc)/(2.0f*thickness_of_pml);
    
    alpha_x      = (float*)malloc(ntx*sizeof(float));
    alpha_x_half = (float*)malloc(ntx*sizeof(float));
   
    alpha_z      = (float*)malloc(ntz*sizeof(float));
    alpha_z_half = (float*)malloc(ntz*sizeof(float));

  //--------------------initialize the vectors--------------
  
  for(ix=0;ix<ntx;ix++)
  {
    a_x[ix]          = 0.0f;
    a_x_half[ix]     = 0.0f;
    b_x[ix]          = 0.0f;
    b_x_half[ix]     = 0.0f;
    d_x[ix]          = 0.0f;
    d_x_half[ix]     = 0.0f;
    k_x[ix]          = 1.0f;
    k_x_half[ix]     = 1.0f;
    alpha_x[ix]      = 0.0f;
    alpha_x_half[ix] = 0.0f;
  }
  
  for(iz=0;iz<ntz;iz++)
  {
    a_z[iz]          = 0.0f;
    a_z_half[iz]     = 0.0f;
    b_z[iz]          = 0.0f;
    b_z_half[iz]     = 0.0f;
    d_z[iz]          = 0.0f;
    d_z_half[iz]     = 0.0f;
    k_z[iz]          = 1.0f;
    k_z_half[iz]     = 1.0f;

    alpha_z[iz]      = 0.0f;
    alpha_z_half[iz] = 0.0f;
  }
  
  
// X direction

  x_start=pml*dx;
  x_end=(ntx-pml-1)*dx;
  
  // Integer points
  for(ix=0;ix<ntx;ix++)
  { 
    x_current=ix*dx;
    
    // LEFT EDGE
    if(x_current<=x_start)
    {
      delta_x=x_start-x_current;
      d_x[ix]=d0*pow(delta_x/thickness_of_pml,2);
      k_x[ix]=1.0f;
      alpha_x[ix]=alpha_max*(1.0f-(delta_x/thickness_of_pml))+0.1f*alpha_max;
    }
    
    // RIGHT EDGE      
    if(x_current>=x_end)
    {
      delta_x=x_current-x_end;
      d_x[ix]=d0*pow(delta_x/thickness_of_pml,2);
      k_x[ix]=1.0f;
      alpha_x[ix]=alpha_max*(1.0f-(delta_x/thickness_of_pml))+0.1f*alpha_max;
    }
  }

  
  // Half Integer points
  for(ix=0;ix<ntx;ix++)
  {
    x_current=(ix+0.5f)*dx;
    
    if(x_current<=x_start)
    {
      delta_x=x_start-x_current;
      d_x_half[ix]=d0*pow(delta_x/thickness_of_pml,2);
      k_x_half[ix]=1.0f;
      alpha_x_half[ix]=alpha_max*(1.0f-(delta_x/thickness_of_pml))+0.1f*alpha_max;
    }
    
    if(x_current>=x_end)
    {
      delta_x=x_current-x_end;
      d_x_half[ix]=d0*pow(delta_x/thickness_of_pml,2);
      k_x_half[ix]=1.0f;
      alpha_x_half[ix]=alpha_max*(1.0f-(delta_x/thickness_of_pml))+0.1f*alpha_max;
    }
  }
  
  for (ix=0;ix<ntx;ix++)
  {
    if(alpha_x[ix]<0.0f)
    {
      alpha_x[ix]=0.0f;
    }
    if(alpha_x_half[ix]<0.0f)
    {
      alpha_x_half[ix]=0.0f;
    }
    
    b_x[ix]=exp(-(d_x[ix]/k_x[ix]+alpha_x[ix])*dt);

    if(d_x[ix] > 1.0e-6f)
    {
      a_x[ix]=d_x[ix]/(k_x[ix]*(d_x[ix]+k_x[ix]*alpha_x[ix]))*(b_x[ix]-1.0f);
    }

    b_x_half[ix]=exp(-(d_x_half[ix]/k_x_half[ix]+alpha_x_half[ix])*dt);
   
    if(d_x_half[ix] > 1.0e-6f)
    {
      a_x_half[ix]=d_x_half[ix]/(k_x_half[ix]*(d_x_half[ix]+k_x_half[ix]*alpha_x_half[ix]))*(b_x_half[ix]-1.0f);
    }
  }
 
// Z direction

  z_start=pml*dx;
  z_end=(ntz-pml-1)*dx;
  
  // Integer points
  for(iz=0;iz<ntz;iz++)
  { 
    z_current=iz*dx;
    
    // LEFT EDGE
    if(z_current<=z_start)
    {
      delta_z=z_start-z_current;
      d_z[iz]=d0*pow(delta_z/thickness_of_pml,2);
      k_z[iz]=1.0f;
      alpha_z[iz]=alpha_max*(1.0f-(delta_z/thickness_of_pml))+0.1f*alpha_max;
    }
    
    // RIGHT EDGE      
    if(z_current>=z_end)
    {
      delta_z=z_current-z_end;
      d_z[iz]=d0*pow(delta_z/thickness_of_pml,2);
      k_z[iz]=1.0f;
      alpha_z[iz]=alpha_max*(1.0f-(delta_z/thickness_of_pml))+0.1f*alpha_max;
    }
  }
  
  // Half Integer points
  for(iz=0;iz<ntz;iz++)
  {
    z_current=(iz+0.5f)*dx;
    
    if(z_current<=z_start)
    {
      delta_z=z_start-z_current;
      d_z_half[iz]=d0*pow(delta_z/thickness_of_pml,2);
      k_z_half[iz]=1.0f;
      alpha_z_half[iz]=alpha_max*(1.0f-(delta_z/thickness_of_pml))+0.1f*alpha_max;
    }
    
    if(z_current>=z_end)
    {
      delta_z=z_current-z_end;
      d_z_half[iz]=d0*pow(delta_z/thickness_of_pml,2);
      k_z_half[iz]=1.0f;
      alpha_z_half[iz]=alpha_max*(1.0f-(delta_z/thickness_of_pml))+0.1f*alpha_max;
    }
  }
  
  for (iz=0;iz<ntz;iz++)
  {
    if(alpha_z[iz]<0.0f)
    {
      alpha_z[iz]=0.0f;
    }
    if(alpha_z_half[iz]<0.0f)
    {
      alpha_z_half[iz]=0.0f;
    }
    
    b_z[iz]=exp(-(d_z[iz]/k_z[iz]+alpha_z[iz])*dt);

    if(d_z[iz]>1.0e-6f)
    {
      a_z[iz]=d_z[iz]/(k_z[iz]*(d_z[iz]+k_z[iz]*alpha_z[iz]))*(b_z[iz]-1.0f);
    }

    b_z_half[iz]=exp(-(d_z_half[iz]/k_z_half[iz]+alpha_z_half[iz])*dt);
    
    if(d_z_half[iz]>1.0e-6f)
    {
      a_z_half[iz]=d_z_half[iz]/(k_z_half[iz]*(d_z_half[iz]+k_z_half[iz]*alpha_z_half[iz]))*(b_z_half[iz]-1.0f);
    }
  }
 
    free(alpha_x);
    free(alpha_x_half);
    free(alpha_z);
    free(alpha_z_half);
    
    return;
    
  }


/*==========================================================
  This subroutine is used for initializing the true model...
===========================================================*/

  void get_acc_model(float *vp, float *vs, float *rho, int ntp, int ntx, int ntz)
  {
    int ip,iz,ix;


    float model_temp[ntz][ntx];
//    int pml=20;
    
    FILE *fp;

    

    fp=fopen("./input/acc_vp_cut.dat","rb");
    for(iz=0;iz<ntz;iz++)
    {
        fread(&model_temp[iz][0],sizeof(float),ntx,fp);
    }
    fclose(fp);

    for(iz=0;iz<ntz;iz++)
    {
        for(ix=0;ix<ntx;ix++)
        {
           ip=iz*ntx+ix;
           vp[ip]=model_temp[iz][ix];
        }
    }

    fp=fopen("./input/acc_vs_cut.dat","rb");
    for(iz=0;iz<ntz;iz++)
    {
        fread(&model_temp[iz][0],sizeof(float),ntx,fp);
    }
    fclose(fp);

    for(iz=0;iz<ntz;iz++)
    {
        for(ix=0;ix<ntx;ix++)
        {
           ip=iz*ntx+ix;
           vs[ip]=model_temp[iz][ix];
        }
    }


    fp=fopen("./input/acc_rho_cut.dat","rb");
    for(iz=0;iz<ntz;iz++)
    {
        fread(&model_temp[iz][0],sizeof(float),ntx,fp);
    }
    fclose(fp);

    for(iz=0;iz<ntz;iz++)
    {
        for(ix=0;ix<ntx;ix++)
        {
           ip=iz*ntx+ix;
           rho[ip]=model_temp[iz][ix];
        }
    }



    return;
  }


/*==========================================================
  This subroutine is used for initializing the initial model...
===========================================================*/

  void get_ini_model(float *vp, float *vs, float *rho, 
                     float *vp_n, float *vs_n,
                     int ntp, int ntx, int ntz)
  {
    int ip,ix,iz;
    FILE *fp;
    float model_temp[ntz][ntx];

    fp=fopen("./input/ini_vp_cut.dat","rb");
    for(iz=0;iz<ntz;iz++)
    {
        fread(&model_temp[iz][0],sizeof(float),ntx,fp);
    }
    fclose(fp);

    for(iz=0;iz<ntz;iz++)
    {
        for(ix=0;ix<ntx;ix++)
        {
           ip=iz*ntx+ix;
           vp[ip]=model_temp[iz][ix];
        }
    }



/*
//  Model in PML..............

    for(iz=0;iz<=pml-1;iz++)
    {

        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+pml;

            vp[ip]=vp[ipp];
        }

        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+ix;
           
            vp[ip]=vp[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+ntx-pml-1;

            vp[ip]=vp[ipp];
        }
    }

    for(iz=pml;iz<=ntz-pml-1;iz++)
    {
        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=iz*ntx+pml;
            
            vp[ip]=vp[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=iz*ntx+ntx-pml-1;
       
            vp[ip]=vp[ipp];
        }

     }

     for(iz=ntz-pml;iz<ntz;iz++)
     {
         
        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+pml;

            vp[ip]=vp[ipp];
        }

        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+ix;
           
            vp[ip]=vp[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+ntx-pml-1;

            vp[ip]=vp[ipp];
        }
    }
*/
    for(ip=0;ip<ntp;ip++)
    {
        vp_n[ip]=vp[ip];
    }

    return;
  }


/*==========================================================
  This subroutine is used for smoothing the  model...
===========================================================*/

  void smooth_model(float *vp, int ntp, int ntx, int ntz, int pml)
  {
/*  flag == 1 :: P velocity
    flag == 2 :: S velocity
    flag == 3 :: Density
*/
    int window=1;
    float *vp_old1;

    float sum;
    int number;

    int iz,ix;
    int izw,ixw;
    int ip;

    vp_old1=(float*)malloc(sizeof(float)*ntp);


    for(ip=0;ip<ntp;ip++)
    {
        vp_old1[ip]=vp[ip];
    }

//-----smooth in the x direction---------
    
    for(iz=pml;iz<=ntz-pml-1;iz++)
    {
        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            sum=0.0f;
            number=0;
            for(izw=iz-window;izw<iz+window;izw++)
            {
                for(ixw=ix-window;ixw<ix+window;ixw++)
                {

                    ip=izw*ntx+ixw;
                    sum=sum+vp_old1[ip];
                    number=number+1;
                }
            }
            ip=iz*ntx+ix;
            vp[ip]=sum/number;
         }
    }    

    free(vp_old1);

    return;
  }

/*==========================================================
  This subroutine is used for finding the maximum value of 
  a vector.
===========================================================*/ 
  void maximum_vector(float *vector, int n, float *maximum_value)
  {
    int i;

    *maximum_value=1.0e-20f;
    for(i=0;i<n;i++)
    {
       if(abs(vector[i])>*maximum_value);
       {
          *maximum_value=abs(vector[i]);
       }
    }
    printf("maximum_value=%f\n",*maximum_value);
    return;
  }


/*==========================================================
  This subroutine is used for finding the minimum value of 
  a vector.
===========================================================*/ 
  void minimum_vector(float *vector, int n, float *minimum_value)
  {
    int i;

    *minimum_value=1.0e+20f;
    for(i=0;i<n;i++)
    {
       if(vector[i]<*minimum_value);
       {
          *minimum_value=vector[i];
       }
    }
    printf("minimum_value=%f\n",*minimum_value);
    return;
  }
  
  

/*==========================================================
  This subroutine is used for calculating the Lame constants...
===========================================================*/

  void get_lame_constants(float *lambda, float *mu, 
       float *lambda_plus_two_mu, float *vp, 
       float * vs, float * rho, int ntp)
  {
    int ip;
    
    // Lambda_plus_two_mu
    for(ip=0;ip<ntp;ip++)
    {
        lambda_plus_two_mu[ip]=vp[ip]*vp[ip]
                                *2.0f*rho[ip];
    }
    
    // Mu
    for(ip=0;ip<ntp;ip++)
    {
        mu[ip]=vs[ip]*vs[ip]
                *2.0f*rho[ip];
    }
    
    // Lambda
    for(ip=0;ip<ntp;ip++)
    {
       lambda[ip]=lambda_plus_two_mu[ip]-2.0f*mu[ip];
    }
    return;
  }

/*==========================================================
  This subroutine is used for calculating the sum of two 
  vectors!
===========================================================*/

  void add(float *a,float *b,float *c,int n)
  {
    int i;
    for(i=0;i<n;i++)
    {
      c[i]=a[i]-b[i];
    }
    
  }
  
/*==========================================================

  This subroutine is used for calculating the ricker wave
   
===========================================================*/
  
  void ricker_wave(float *rick, int itmax, float f0, float t0, float dt)
  {
    float pi=3.1415927f;
    int   it;
    float temp;

    FILE *fp;
    
    for(it=0;it<itmax;it++)
    {
      temp=pi*f0*(it*dt-t0);
      temp=temp*temp;
      rick[it]=(1.0f-2.0f*temp)*exp(-temp);
    }

    fp=fopen("./output/rick.dat","w");
    
    for(it=0;it<itmax;it++)
    {
      fprintf(fp,"%18f%18f\n",it*dt,rick[it]);
    }
    
    fclose(fp);

  }

/*==========================================================

  This subroutine is used for calculating the forward wave 
  field of 2D in time domain.

1.
  inv_flag==0----Calculate the observed seismograms of 
                 Vx and Vz components...
2.
  inv_flag==1----Calculate the synthetic seismograms of 
                 Vx and Vz components and store the 
                 borders of Vx and Vz used for constructing 
                 the forward wavefields. 
===========================================================*/

  void fdtd_cpml_2d_GPU(int ntx, int ntz, int ntp, int nx, int nz,
                    int pml, float dx, float dz,
                    float *rick, int itmax, float dt,
                    int is, int s_ix, int s_iz, float *rho,
                    float *lambda, float *mu, float *lambda_plus_two_mu,
                    float *k_x, float *k_x_half,
                    float *k_z, float *k_z_half,
                    float *a_x, float *a_x_half,
                    float *a_z, float *a_z_half,
                    float *b_x, float *b_x_half,
                    float *b_z, float *b_z_half, 
                    float *c, int inv_flag,
                    float *seismogram_vx, float *seismogram_vz,
                    float *vx_borders_up, float *vx_borders_bottom,
                    float *vx_borders_left, float *vx_borders_right,
                    float *vz_borders_up, float *vz_borders_bottom,
                    float *vz_borders_left, float *vz_borders_right)
  {
    int it,ip;
  
    float *vx,*vz;
    float *sigmaxx,*sigmaxz,*sigmazz;
  
    float *phi_vx_x,*phi_vx_z,*phi_vz_z,*phi_vz_x;
  
    float *phi_sigmaxx_x,*phi_sigmaxz_z;
    float *phi_sigmaxz_x,*phi_sigmazz_z;

    char filename[230];
    FILE *fp;

    // vectors for the devices

    float *d_rick;
    float *d_lambda, *d_mu, *d_rho;
    float *d_lambda_plus_two_mu;

//    float *d_k_x, *d_k_x_half;
//    float *d_k_z, *d_k_z_half;

    float *d_a_x, *d_a_x_half;
    float *d_a_z, *d_a_z_half;
    float *d_b_x, *d_b_x_half;
    float *d_b_z, *d_b_z_half;

    float *d_vx,*d_vz;
    float *d_sigmaxx,*d_sigmaxz,*d_sigmazz;
  
    float *d_phi_vx_x,*d_phi_vx_z,*d_phi_vz_z,*d_phi_vz_x;
  
    float *d_phi_sigmaxx_x,*d_phi_sigmaxz_z;
    float *d_phi_sigmaxz_x,*d_phi_sigmazz_z;

    float *d_c;

    float *snap,*d_snap;

    size_t size_model=sizeof(float)*ntp;

//    float *seismogram_vx,*seismogram_vz;

    float *d_seismogram_vx,*d_seismogram_vz;

    float *d_vx_borders_up,*d_vx_borders_bottom;
    float *d_vx_borders_left,*d_vx_borders_right;

    float *d_vz_borders_up,*d_vz_borders_bottom;
    float *d_vz_borders_left,*d_vz_borders_right;


    int receiver_z=pml+5;

    // allocate the memory of recording of Vx,Vz........  
  
    snap=(float*)malloc(sizeof(float)*nx*nz); 

    // allocate the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,...

    vx=(float*)malloc(sizeof(float)*ntp); 
    vz=(float*)malloc(sizeof(float)*ntp); 
    sigmaxx=(float*)malloc(sizeof(float)*ntp);
    sigmazz=(float*)malloc(sizeof(float)*ntp);
    sigmaxz=(float*)malloc(sizeof(float)*ntp);

    // allocate the memory of phi_vx_x...

    phi_vx_x      = (float*)malloc(sizeof(float)*ntp);
    phi_vz_z      = (float*)malloc(sizeof(float)*ntp);
    phi_vx_z      = (float*)malloc(sizeof(float)*ntp);
    phi_vz_x      = (float*)malloc(sizeof(float)*ntp); 
    
    // allocate the memory of phi_sigmaxx_x...

    phi_sigmaxx_x=(float*)malloc(sizeof(float)*ntp);
    phi_sigmaxz_z=(float*)malloc(sizeof(float)*ntp);
    phi_sigmaxz_x=(float*)malloc(sizeof(float)*ntp);
    phi_sigmazz_z=(float*)malloc(sizeof(float)*ntp);

    // allocate the memory for the device

    cudaMalloc((void**)&d_seismogram_vx,sizeof(float)*itmax*nx);
    cudaMalloc((void**)&d_seismogram_vz,sizeof(float)*itmax*nx);

    cudaMalloc((void**)&d_vx_borders_up,sizeof(float)*itmax*(nx+1));
    cudaMalloc((void**)&d_vx_borders_bottom,sizeof(float)*itmax*(nx+1));
    cudaMalloc((void**)&d_vx_borders_left,sizeof(float)*itmax*(nz-2));
    cudaMalloc((void**)&d_vx_borders_right,sizeof(float)*itmax*(nz-2));

    cudaMalloc((void**)&d_vz_borders_up,sizeof(float)*itmax*(nx));
    cudaMalloc((void**)&d_vz_borders_bottom,sizeof(float)*itmax*(nx));
    cudaMalloc((void**)&d_vz_borders_left,sizeof(float)*itmax*(nz-1));
    cudaMalloc((void**)&d_vz_borders_right,sizeof(float)*itmax*(nz-1));

    cudaMalloc((void**)&d_snap,sizeof(float)*nx*nz);
    cudaMalloc((void**)&d_c,sizeof(float)*2);
    cudaMalloc((void**)&d_rick,sizeof(float)*itmax);        // ricker wave 

    cudaMalloc((void**)&d_lambda,size_model);
    cudaMalloc((void**)&d_mu,size_model);
    cudaMalloc((void**)&d_rho,size_model);
    cudaMalloc((void**)&d_lambda_plus_two_mu,size_model);   // model 
     

    cudaMalloc((void**)&d_a_x,sizeof(float)*ntx);
    cudaMalloc((void**)&d_a_x_half,sizeof(float)*ntx);
    cudaMalloc((void**)&d_a_z,sizeof(float)*ntz);
    cudaMalloc((void**)&d_a_z_half,sizeof(float)*ntz);

    cudaMalloc((void**)&d_b_x,sizeof(float)*ntx);
    cudaMalloc((void**)&d_b_x_half,sizeof(float)*ntx);
    cudaMalloc((void**)&d_b_z,sizeof(float)*ntz);
    cudaMalloc((void**)&d_b_z_half,sizeof(float)*ntz);      // atten parameters


    cudaMalloc((void**)&d_vx,size_model);
    cudaMalloc((void**)&d_vz,size_model);
    cudaMalloc((void**)&d_sigmaxx,size_model);
    cudaMalloc((void**)&d_sigmazz,size_model);
    cudaMalloc((void**)&d_sigmaxz,size_model);              // wavefields 


    cudaMalloc((void**)&d_phi_vx_x,size_model);
    cudaMalloc((void**)&d_phi_vz_z,size_model);
    cudaMalloc((void**)&d_phi_vx_z,size_model);
    cudaMalloc((void**)&d_phi_vz_x,size_model);

    cudaMalloc((void**)&d_phi_sigmaxx_x,size_model);
    cudaMalloc((void**)&d_phi_sigmaxz_z,size_model);
    cudaMalloc((void**)&d_phi_sigmaxz_x,size_model);
    cudaMalloc((void**)&d_phi_sigmazz_z,size_model);
    
    // Initialize the fields........................
    
    for(ip=0;ip<nx*nz;ip++)
    {
        snap[ip]=0.0f;
    }
    for(ip=0;ip<ntp;ip++)
    {
        vx[ip]=0.0f;
        vz[ip]=0.0f;
           	
        sigmaxx[ip]=0.0f;
        sigmazz[ip]=0.0f;
        sigmaxz[ip]=0.0f;
          
        phi_vx_x[ip]=0.0f;
        phi_vz_z[ip]=0.0f;
        phi_vx_z[ip]=0.0f;
        phi_vz_x[ip]=0.0f;

        phi_sigmaxx_x[ip]=0.0f;
        phi_sigmaxz_z[ip]=0.0f;
    
        phi_sigmaxz_x[ip]=0.0f;
        phi_sigmazz_z[ip]=0.0f;
    }

    for(ip=0;ip<itmax*nx;ip++)
    {
       seismogram_vx[ip]=0.0f;
       seismogram_vz[ip]=0.0f;
    }

    // Copy the vectors from the host to the device
    cudaMemcpy(d_seismogram_vx,seismogram_vx,sizeof(float)*nx*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_seismogram_vz,seismogram_vz,sizeof(float)*nx*itmax,cudaMemcpyHostToDevice);

    cudaMemcpy(d_vx_borders_up,vx_borders_up,sizeof(float)*(nx+1)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx_borders_bottom,vx_borders_bottom,sizeof(float)*(nx+1)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx_borders_left,vx_borders_left,sizeof(float)*(nz-2)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx_borders_right,vx_borders_right,sizeof(float)*(nz-2)*itmax,cudaMemcpyHostToDevice);

    cudaMemcpy(d_vz_borders_up,vz_borders_up,sizeof(float)*(nx)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz_borders_bottom,vz_borders_bottom,sizeof(float)*(nx)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz_borders_left,vz_borders_left,sizeof(float)*(nz-1)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz_borders_right,vz_borders_right,sizeof(float)*(nz-1)*itmax,cudaMemcpyHostToDevice);


    cudaMemcpy(d_snap,snap,sizeof(float)*nx*nz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_c,c,sizeof(float)*2,cudaMemcpyHostToDevice);
    cudaMemcpy(d_rick,rick,sizeof(float)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda,lambda,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu,mu,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda_plus_two_mu,lambda_plus_two_mu,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho,rho,size_model,cudaMemcpyHostToDevice);

    cudaMemcpy(d_a_x,a_x,sizeof(float)*ntx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_x_half,a_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_z,a_z,sizeof(float)*ntz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_z_half,a_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice);

    cudaMemcpy(d_b_x,b_x,sizeof(float)*ntx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_x_half,b_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_z,b_z,sizeof(float)*ntz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_z_half,b_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice);

    cudaMemcpy(d_vx,vx,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz,vz,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmaxx,sigmaxx,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmazz,sigmazz,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmaxz,sigmaxz,size_model,cudaMemcpyHostToDevice);

    cudaMemcpy(d_phi_vx_x,phi_vx_x,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_vz_z,phi_vz_z,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_vx_z,phi_vx_z,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_vz_x,phi_vz_x,size_model,cudaMemcpyHostToDevice);

    cudaMemcpy(d_phi_sigmaxx_x,phi_sigmaxx_x,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_sigmaxz_z,phi_sigmaxz_z,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_sigmaxz_x,phi_sigmaxz_x,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_sigmazz_z,phi_sigmazz_z,size_model,cudaMemcpyHostToDevice);
    
// =============================================================================

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(ntz+dimBlock.y-1)/dimBlock.y);

//-----------------------------------------------------------------------//
//=======================================================================//
//-----------------------------------------------------------------------//
    for(it=0;it<itmax;it++)
    {

      if(WITH_SHARED_MEMORY==1)
      {
        fdtd_cpml_2d_GPU_kernel_vx_shared<<<dimGrid,dimBlock>>>
        (
           d_rho, d_a_x_half, d_a_z, 
           d_b_x_half, d_b_z, 
           d_vx, d_sigmaxx, d_sigmaxz,
           d_phi_sigmaxx_x, d_phi_sigmaxz_z, 
           ntp, ntx, ntz, dx, dz, dt,
           d_snap
        );

        fdtd_cpml_2d_GPU_kernel_vz_shared<<<dimGrid,dimBlock>>>
        (
           d_rho,
           d_a_x, d_a_z_half,
           d_b_x, d_b_z_half,
           d_vz, d_sigmaxz, d_sigmazz, 
           d_phi_sigmaxz_x, d_phi_sigmazz_z,
           ntp, ntx, ntz, dx, dz, dt
        );

        fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_shared<<<dimGrid,dimBlock>>>
        (
           d_rick, 
           d_lambda, d_lambda_plus_two_mu,
           d_a_x,d_a_z,d_b_x,d_b_z,
           d_vx, d_vz, d_sigmaxx, d_sigmazz,
           d_phi_vx_x,d_phi_vz_z,
           ntp, ntx, ntz, dx, dz, dt,
           s_ix, s_iz, it
        );

        fdtd_cpml_2d_GPU_kernel_sigmaxz_shared<<<dimGrid,dimBlock>>>
        (
           d_mu,
           d_a_x_half, d_a_z_half,
           d_b_x_half, d_b_z_half,
           d_vx, d_vz, d_sigmaxz,
           d_phi_vx_z, d_phi_vz_x,
           ntp, ntx, ntz, dx, dz, dt
        );
      }

      else
      {

        fdtd_cpml_2d_GPU_kernel_vx<<<dimGrid,dimBlock>>>
        (
           d_rho, d_a_x_half, d_a_z, 
           d_b_x_half, d_b_z, 
           d_vx, d_sigmaxx, d_sigmaxz,
           d_phi_sigmaxx_x, d_phi_sigmaxz_z, 
           ntp, ntx, ntz, dx, dz, dt,
           d_seismogram_vx, it, pml,receiver_z,
           inv_flag,
           d_vx_borders_up,d_vx_borders_bottom,
           d_vx_borders_left,d_vx_borders_right
        );

        fdtd_cpml_2d_GPU_kernel_vz<<<dimGrid,dimBlock>>>
        (
           d_rho,
           d_a_x, d_a_z_half,
           d_b_x, d_b_z_half,
           d_vz, d_sigmaxz, d_sigmazz, 
           d_phi_sigmaxz_x, d_phi_sigmazz_z,
           ntp, ntx, ntz, dx, dz, dt,
           d_seismogram_vz, it, pml,receiver_z,
           inv_flag,
           d_vz_borders_up,d_vz_borders_bottom,
           d_vz_borders_left,d_vz_borders_right
        );

        fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz<<<dimGrid,dimBlock>>>
        (
           d_rick, 
           d_lambda, d_lambda_plus_two_mu,
           d_a_x,d_a_z,d_b_x,d_b_z,
           d_vx, d_vz, d_sigmaxx, d_sigmazz,
           d_phi_vx_x,d_phi_vz_z,
           ntp, ntx, ntz, dx, dz, dt,
           s_ix, s_iz, it,
           inv_flag
        );

        fdtd_cpml_2d_GPU_kernel_sigmaxz<<<dimGrid,dimBlock>>>
        (
           d_mu,
           d_a_x_half, d_a_z_half,
           d_b_x_half, d_b_z_half,
           d_vx, d_vz, d_sigmaxz,
           d_phi_vx_z, d_phi_vz_x,
           ntp, ntx, ntz, dx, dz, dt,
           inv_flag
        );


     }


     if(inv_flag==0)

     {

       if(it==itmax-1)
       {
          cudaMemcpy(seismogram_vx,d_seismogram_vx,
                     sizeof(float)*nx*itmax,cudaMemcpyDeviceToHost);
          cudaMemcpy(seismogram_vz,d_seismogram_vz,
                     sizeof(float)*nx*itmax,cudaMemcpyDeviceToHost);

          sprintf(filename,"./output/%dsource_seismogram_vx_obs.dat",is+1);   
     
          fp=fopen(filename,"wb");

          fwrite(&seismogram_vx[0],sizeof(float),nx*itmax,fp);
          fclose(fp);

          sprintf(filename,"./output/%dsource_seismogram_vz_obs.dat",is+1);
          fp=fopen(filename,"wb");

          fwrite(&seismogram_vz[0],sizeof(float),nx*itmax,fp);
          fclose(fp);
       }
     }


     if(inv_flag==1)
     {
       if(it==itmax-1)
       {
          cudaMemcpy(seismogram_vx,d_seismogram_vx,
                     sizeof(float)*nx*itmax,cudaMemcpyDeviceToHost);
          cudaMemcpy(seismogram_vz,d_seismogram_vz,
                     sizeof(float)*nx*itmax,cudaMemcpyDeviceToHost);


          cudaMemcpy(vx_borders_up,d_vx_borders_up,
                     sizeof(float)*(nx+1)*itmax,cudaMemcpyDeviceToHost);
          cudaMemcpy(vx_borders_bottom,d_vx_borders_bottom,
                     sizeof(float)*(nx+1)*itmax,cudaMemcpyDeviceToHost);
          cudaMemcpy(vx_borders_left,d_vx_borders_left,
                     sizeof(float)*(nz-2)*itmax,cudaMemcpyDeviceToHost);
          cudaMemcpy(vx_borders_right,d_vx_borders_right,
                     sizeof(float)*(nz-2)*itmax,cudaMemcpyDeviceToHost);

          cudaMemcpy(vz_borders_up,d_vz_borders_up,
                     sizeof(float)*(nx)*itmax,cudaMemcpyDeviceToHost);
          cudaMemcpy(vz_borders_bottom,d_vz_borders_bottom,
                     sizeof(float)*(nx)*itmax,cudaMemcpyDeviceToHost);
          cudaMemcpy(vz_borders_left,d_vz_borders_left,
                     sizeof(float)*(nz-1)*itmax,cudaMemcpyDeviceToHost);
          cudaMemcpy(vz_borders_right,d_vz_borders_right,
                     sizeof(float)*(nz-1)*itmax,cudaMemcpyDeviceToHost);
       }
     }


     if(inv_flag==1)
     {
       if(it==itmax-1)
       {

          // Output The wavefields when Time=Itmax;

          cudaMemcpy(vx,d_vx,size_model,cudaMemcpyDeviceToHost);
          cudaMemcpy(vz,d_vz,size_model,cudaMemcpyDeviceToHost);
          cudaMemcpy(sigmaxx,d_sigmaxx,size_model,cudaMemcpyDeviceToHost);
          cudaMemcpy(sigmazz,d_sigmazz,size_model,cudaMemcpyDeviceToHost);
          cudaMemcpy(sigmaxz,d_sigmaxz,size_model,cudaMemcpyDeviceToHost);
     
          fp=fopen("./output/wavefield_itmax.dat","wb");
          fwrite(&vx[0],sizeof(float),ntp,fp);
          fwrite(&vz[0],sizeof(float),ntp,fp);

          fwrite(&sigmaxx[0],sizeof(float),ntp,fp);
          fwrite(&sigmazz[0],sizeof(float),ntp,fp);
          fwrite(&sigmaxz[0],sizeof(float),ntp,fp);
          fclose(fp);

       }
    }

     if(inv_flag==3)
     {
       if(it==itmax-1)
       {
          cudaMemcpy(seismogram_vx,d_seismogram_vx,
                     sizeof(float)*nx*itmax,cudaMemcpyDeviceToHost);
          cudaMemcpy(seismogram_vz,d_seismogram_vz,
                     sizeof(float)*nx*itmax,cudaMemcpyDeviceToHost);
       }
     }


     if(OUTPUT_SNAP==1)
     { 
        if(inv_flag==1)
        {
           if(it%10==0)
           {
              cudaMemcpy(vx,d_vx,sizeof(float)*ntp,cudaMemcpyDeviceToHost);

              sprintf(filename,"./output/%dvx_for.dat",it);   
     
              fp=fopen(filename,"wb");
              fwrite(&vx[0],sizeof(float),ntp,fp);
              fclose(fp);

           }
        }
     }

    }


    //free the memory of recording of vx,vz;
//    free(seismogram_vx);
//    free(seismogram_vz);
    free(snap);

    //free the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,Sigmazz...  
    free(vx);
    free(vz);
    free(sigmaxx);
    free(sigmazz);
    free(sigmaxz);
    
   //free the memory of Phi_vx_x....  
    free(phi_vx_x);
    free(phi_vz_z);
    free(phi_vx_z);
    free(phi_vz_x);
    
    //free the memory of Phi_vx_x....  
    free(phi_sigmaxx_x);
    free(phi_sigmaxz_z);
    free(phi_sigmaxz_x);
    free(phi_sigmazz_z);
    
    //free the memory of DEVICE
    cudaFree(d_seismogram_vx);
    cudaFree(d_seismogram_vz);

    cudaFree(d_vx_borders_up);
    cudaFree(d_vx_borders_bottom);
    cudaFree(d_vx_borders_left);
    cudaFree(d_vx_borders_right);

    cudaFree(d_vz_borders_up);
    cudaFree(d_vz_borders_bottom);
    cudaFree(d_vz_borders_left);
    cudaFree(d_vz_borders_right);

    cudaFree(d_snap);
    cudaFree(d_c);
    cudaFree(d_rick);
    cudaFree(d_lambda);
    cudaFree(d_mu);
    cudaFree(d_lambda_plus_two_mu);
    cudaFree(d_rho);
    
    cudaFree(d_a_x);
    cudaFree(d_a_x_half);
    cudaFree(d_a_z);
    cudaFree(d_a_z_half);

    cudaFree(d_b_x);
    cudaFree(d_b_x_half);
    cudaFree(d_b_z);
    cudaFree(d_b_z_half);

    cudaFree(d_vx);
    cudaFree(d_vz);
    cudaFree(d_sigmaxx);
    cudaFree(d_sigmazz);
    cudaFree(d_sigmaxz);
    
    cudaFree(d_phi_vx_x);
    cudaFree(d_phi_vz_z);
    cudaFree(d_phi_vx_z);
    cudaFree(d_phi_vz_x);
      
    cudaFree(d_phi_sigmaxx_x);
    cudaFree(d_phi_sigmaxz_z);
    cudaFree(d_phi_sigmaxz_x);
    cudaFree(d_phi_sigmazz_z);

  }


  __global__ void fdtd_cpml_2d_GPU_kernel_vx(
    float *rho,
    float *a_x_half, float *a_z, 
    float *b_x_half, float *b_z, 
    float *vx, float *sigmaxx, float *sigmaxz,
    float *phi_sigmaxx_x, float *phi_sigmaxz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt,
    float *seismogram_vx, int it, int pml,
    int receiver_z, int inv_flag,
    float *vx_borders_up, float *vx_borders_bottom,
    float *vx_borders_left, float *vx_borders_right
    )

  {

    int nx=ntx-2*pml;
    int nz=ntz-2*pml;

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxx_dx,dsigmaxz_dz;
    float one_over_rho_half_x;

    if(iz>=2&&iz<ntz-1&&ix>=1&&ix<ntx-2)
    {
       ip=iz*ntx+ix;
       dsigmaxx_dx=(c[0]*(sigmaxx[ip+1]-sigmaxx[ip])+
                    c[1]*(sigmaxx[ip+2]-sigmaxx[ip-1]))*one_over_dx;
       dsigmaxz_dz=(c[0]*(sigmaxz[ip]-sigmaxz[ip-ntx])+
                    c[1]*(sigmaxz[ip+ntx]-sigmaxz[ip-2*ntx]))*one_over_dz;
                 
       phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
       phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

       one_over_rho_half_x=1.0f/(0.5f*(rho[ip]+rho[ip+1]));
                 
       vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip]
                                     +dsigmaxz_dz+phi_sigmaxz_z[ip])
                                     +vx[ip];
    }
    __syncthreads();

    

    // Seismogram...   
    if(ix>=pml&&ix<=ntx-pml-1&&iz==receiver_z)
    {
       seismogram_vx[it*(ntx-2*pml)+ix-pml]=vx[ip];
    }
    __syncthreads();

   
    // Borders...
    if(inv_flag==1)
    {
       if(ix>=pml-1&&ix<=ntx-pml-1&&iz==pml)
       {
          vx_borders_up[it*(nx+1)+ix-pml+1]=vx[ip];
       }
       if(ix>=pml-1&&ix<=ntx-pml-1&&iz==ntz-pml-1)
       {
          vx_borders_bottom[it*(nx+1)+ix-pml+1]=vx[ip];
       }


       if(iz>=pml+1&&iz<=ntz-pml-2&&ix==pml-1)
       {
          vx_borders_left[it*(nz-2)+iz-pml-1]=vx[ip];
       }
       if(iz>=pml+1&&iz<=ntz-pml-2&&ix==ntx-pml-1)
       {
          vx_borders_right[it*(nz-2)+iz-pml-1]=vx[ip];
       }
    }
    __syncthreads();

  }

  __global__ void fdtd_cpml_2d_GPU_kernel_vx_shared(
    float *rho,
    float *a_x_half, float *a_z, 
    float *b_x_half, float *b_z, 
    float *vx, float *sigmaxx, float *sigmaxz,
    float *phi_sigmaxx_x, float *phi_sigmaxz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt,
    float *snap
    )
  {

    int radius=2;
    __shared__ float s_sigmaxx[BLOCK_SIZE+4][BLOCK_SIZE+4];
    __shared__ float s_sigmaxz[BLOCK_SIZE+4][BLOCK_SIZE+4];

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int s_tx=tx+radius;  // Thread's x-index into corresponding shared memory tile
    int s_ty=ty+radius;  // Thread's y-index into corresponding shared memory tile

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxx_dx,dsigmaxz_dz;
    float one_over_rho_half_x;

    if(iz>=2&&iz<ntz-1&&ix>=1&&ix<ntx-2)
    {
    ip=ntx*iz+ix;
    if(tx<radius)
    {
       s_sigmaxx[s_ty][tx+BLOCK_SIZE+radius]=sigmaxx[ip+BLOCK_SIZE];
       s_sigmaxx[s_ty][tx]=sigmaxx[ip-radius];
    }
    
    s_sigmaxx[s_ty][s_tx]=sigmaxx[ip];


    if(ty<radius)
    {
       s_sigmaxz[ty+BLOCK_SIZE+radius][s_tx]=sigmaxz[ip+BLOCK_SIZE*ntx];
       s_sigmaxz[ty][s_tx]=sigmaxz[ip-radius*ntx];
    }

    s_sigmaxz[s_ty][s_tx]=sigmaxz[ip];

    __syncthreads();

       dsigmaxx_dx=(c[0]*(s_sigmaxx[s_ty][s_tx+1]-s_sigmaxx[s_ty][s_tx])+
                    c[1]*(s_sigmaxx[s_ty][s_tx+2]-s_sigmaxx[s_ty][s_tx-1]))*one_over_dx;


       dsigmaxz_dz=(c[0]*(s_sigmaxz[s_ty][s_tx]-s_sigmaxz[s_ty-1][s_tx])+
                    c[1]*(s_sigmaxz[s_ty+1][s_tx]-s_sigmaxz[s_ty-2][s_tx]))*one_over_dz;


                 
       phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
       phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

       one_over_rho_half_x=1.0f/(0.5f*(rho[ip]+rho[ip+1]));
                 
       vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip]
                                     +dsigmaxz_dz+phi_sigmaxz_z[ip])
                                     +vx[ip];
    }

    __syncthreads();

  }


  __global__ void fdtd_cpml_2d_GPU_kernel_vz(
    float *rho,
    float *a_x, float *a_z_half,
    float *b_x, float *b_z_half,
    float *vz, float *sigmaxz, float *sigmazz, 
    float *phi_sigmaxz_x, float *phi_sigmazz_z,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt,
    float *seismogram_vz, int it, int pml,
    int receiver_z, int inv_flag,
    float *vz_borders_up, float *vz_borders_bottom,
    float *vz_borders_left, float *vz_borders_right
    )
  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxz_dx,dsigmazz_dz;

    float one_over_rho_half_z;

    int ip;
    int nx=ntx-2*pml;
    int nz=ntz-2*pml;

    if(iz>=1&&iz<ntz-2&&ix>=2&&ix<ntx-1)
    {
       ip=iz*ntx+ix;
       dsigmaxz_dx=(c[0]*(sigmaxz[ip]-sigmaxz[ip-1])+
                    c[1]*(sigmaxz[ip+1]-sigmaxz[ip-2]))*one_over_dx;
       dsigmazz_dz=(c[0]*(sigmazz[ip+ntx]-sigmazz[ip])+
                    c[1]*(sigmazz[ip+2*ntx]-sigmazz[ip-ntx]))*one_over_dz;
                 
       phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;
       phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;


       one_over_rho_half_z=1.0f/(0.5f*(rho[ip]+rho[ip+ntx]));
                 
       vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip]
                                     +dsigmazz_dz+phi_sigmazz_z[ip])
                                     +vz[ip];
    }
    __syncthreads();

    if(ix>=pml&&ix<=ntx-pml-1&&iz==receiver_z)
    {
       seismogram_vz[it*(ntx-2*pml)+ix-pml]=vz[ip];
    }
    __syncthreads();

    // Borders...
    if(inv_flag==1)
    {

       if(ix>=pml&&ix<=ntx-pml-1&&iz==pml-1)
       {
          vz_borders_up[it*(nx)+ix-pml]=vz[ip];
       }
       if(ix>=pml&&ix<=ntx-pml-1&&iz==ntz-pml-1)
       {
          vz_borders_bottom[it*(nx)+ix-pml]=vz[ip];
       }

       if(iz>=pml&&iz<=ntz-pml-2&&ix==pml)
       {
          vz_borders_left[it*(nz-1)+iz-pml]=vz[ip];
       }
       if(iz>=pml&&iz<=ntz-pml-2&&ix==ntx-pml-1)
       {
          vz_borders_right[it*(nz-1)+iz-pml]=vz[ip];
       }
    }
    __syncthreads();

  }

  __global__ void fdtd_cpml_2d_GPU_kernel_vz_shared(
    float *rho,
    float *a_x, float *a_z_half,
    float *b_x, float *b_z_half,
    float *vz, float *sigmaxz, float *sigmazz, 
    float *phi_sigmaxz_x, float *phi_sigmazz_z,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    )
  {
    int radius=2;
    __shared__ float s_sigmaxz[BLOCK_SIZE+4][BLOCK_SIZE+4];
    __shared__ float s_sigmazz[BLOCK_SIZE+4][BLOCK_SIZE+4];

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int s_tx=tx+radius;  // Thread's x-index into corresponding shared memory tile
    int s_ty=ty+radius;  // Thread's y-index into corresponding shared memory tile

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxz_dx,dsigmazz_dz;

    float one_over_rho_half_z;

    int ip;


    if(iz>=1&&iz<ntz-2&&ix>=2&&ix<ntx-1)
    {
    ip=ntx*iz+ix;
    if(tx<radius)
    {
       s_sigmaxz[s_ty][tx+BLOCK_SIZE+radius]=sigmaxz[ip+BLOCK_SIZE];
       s_sigmaxz[s_ty][tx]=sigmaxz[ip-radius];
    }
    
    s_sigmaxz[s_ty][s_tx]=sigmaxz[ip];


    if(ty<radius)
    {
       s_sigmazz[ty+BLOCK_SIZE+radius][s_tx]=sigmazz[ip+BLOCK_SIZE*ntx];
       s_sigmazz[ty][s_tx]=sigmazz[ip-radius*ntx];
    }

    s_sigmazz[s_ty][s_tx]=sigmazz[ip];

    __syncthreads();

       dsigmaxz_dx=(c[0]*(s_sigmaxz[s_ty][s_tx]-s_sigmaxz[s_ty][s_tx-1])+
                    c[1]*(s_sigmaxz[s_ty][s_tx+1]-s_sigmaxz[s_ty][s_tx-2]))*one_over_dx;
       dsigmazz_dz=(c[0]*(s_sigmazz[s_ty+1][s_tx]-s_sigmazz[s_ty][s_tx])+
                    c[1]*(s_sigmazz[s_ty+2][s_tx]-s_sigmazz[s_ty-1][s_tx]))*one_over_dz;
                 
       phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;
       phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;


       one_over_rho_half_z=1.0f/(0.5f*(rho[ip]+rho[ip+ntx]));
                 
       vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip]
                                     +dsigmazz_dz+phi_sigmazz_z[ip])
                                     +vz[ip];
    }
    __syncthreads();
  }


  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz(
    float *rick, 
    float *lambda, float *lambda_plus_two_mu,
    float *a_x, float *a_z,
    float *b_x, float *b_z,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    float *phi_vx_x, float *phi_vz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt, int s_ix, int s_iz, int it,
    int inv_flag
    )

  {
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dvx_dx,dvz_dz;
    int ip;



    if(iz>=2&&iz<ntz-1&&ix>=2&&ix<ntx-1)
    {
       ip=iz*ntx+ix;
       dvx_dx=(c[0]*(vx[ip]-vx[ip-1])+
               c[1]*(vx[ip+1]-vx[ip-2]))*one_over_dx;
       dvz_dz=(c[0]*(vz[ip]-vz[ip-ntx])+
               c[1]*(vz[ip+ntx]-vz[ip-2*ntx]))*one_over_dz;
            
       phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
       phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;

            
       sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx+phi_vx_x[ip])+
                    lambda[ip]*(dvz_dz+phi_vz_z[ip]))*dt+
                    sigmaxx[ip];

       sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz+phi_vz_z[ip])+
                    lambda[ip]*(dvx_dx+phi_vx_x[ip]))*dt+
                    sigmazz[ip];
    }

     if(iz==s_iz&&ix==s_ix)
     {
        sigmaxx[ip]=sigmaxx[ip]+rick[it];
        sigmazz[ip]=sigmazz[ip]+rick[it];
     }

    __syncthreads();
  }

  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_shared(
    float *rick, 
    float *lambda, float *lambda_plus_two_mu,
    float *a_x, float *a_z,
    float *b_x, float *b_z,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    float *phi_vx_x, float *phi_vz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt, int s_ix, int s_iz, int it
    )

  {

    int radius=2;
    __shared__ float s_vx[BLOCK_SIZE+4][BLOCK_SIZE+4];
    __shared__ float s_vz[BLOCK_SIZE+4][BLOCK_SIZE+4];

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int s_tx=tx+radius;  // Thread's x-index into corresponding shared memory tile
    int s_ty=ty+radius;  // Thread's y-index into corresponding shared memory tile


    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dvx_dx,dvz_dz;
    int ip;


    if(iz>=2&&iz<ntz-1&&ix>=2&&ix<ntx-1)
    {
    ip=ntx*iz+ix;
    if(tx<radius)
    {
       s_vx[s_ty][tx+BLOCK_SIZE+radius]=vx[ip+BLOCK_SIZE];
       s_vx[s_ty][tx]=vx[ip-radius];
    }
    
    s_vx[s_ty][s_tx]=vx[ip];


    if(ty<radius)
    {
       s_vz[ty+BLOCK_SIZE+radius][s_tx]=vz[ip+BLOCK_SIZE*ntx];
       s_vz[ty][s_tx]=vz[ip-radius*ntx];
    }

    s_vz[s_ty][s_tx]=vz[ip];

    __syncthreads();

       dvx_dx=(c[0]*(s_vx[s_ty][s_tx]-s_vx[s_ty][s_tx-1])+
               c[1]*(s_vx[s_ty][s_tx+1]-s_vx[s_ty][s_tx-2]))*one_over_dx;
       dvz_dz=(c[0]*(s_vz[s_ty][s_tx]-s_vz[s_ty-1][s_tx])+
               c[1]*(s_vz[s_ty+1][s_tx]-s_vz[s_ty-2][s_tx]))*one_over_dz;
            
       phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
       phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;

            
       sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx+phi_vx_x[ip])+
                    lambda[ip]*(dvz_dz+phi_vz_z[ip]))*dt+
                    sigmaxx[ip];

       sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz+phi_vz_z[ip])+
                    lambda[ip]*(dvx_dx+phi_vx_x[ip]))*dt+
                    sigmazz[ip];
    }

     if(iz==s_iz&&ix==s_ix)
     {
        sigmaxx[ip]=sigmaxx[ip]+rick[it];
        sigmazz[ip]=sigmazz[ip]+rick[it];
     }

    __syncthreads();
  }


  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz(
    float *mu,
    float *a_x_half, float *a_z_half,
    float *b_x_half, float *b_z_half,
    float *vx, float *vz, float *sigmaxz,
    float *phi_vx_z, float *phi_vz_x,
    int ntp, int ntx, int ntz,
    float dx, float dz, float dt,
    int inv_flag
    )
  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;
    float dvx_dz,dvz_dx;
    float mu_half_x_half_z;

    int ip;

     if(iz>=1&&iz<ntz-2&&ix>=1&&ix<ntx-2)
     {
       ip=iz*ntx+ix;      
        dvz_dx=(c[0]*(vz[ip+1]-vz[ip])+
                c[1]*(vz[ip+2]-vz[ip-1]))*one_over_dx;
        dvx_dz=(c[0]*(vx[ip+ntx]-vx[ip])+
                c[1]*(vx[ip+2*ntx]-vx[ip-ntx]))*one_over_dz;
              
        phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
        phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;

  
        mu_half_x_half_z=0.25f*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);
 
        sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+phi_vz_x[ip]+
                                      dvx_dz+phi_vx_z[ip])*dt+
                                      sigmaxz[ip];
     }

     __syncthreads();
  }


  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_shared(
    float *mu,
    float *a_x_half, float *a_z_half,
    float *b_x_half, float *b_z_half,
    float *vx, float *vz, float *sigmaxz,
    float *phi_vx_z, float *phi_vz_x,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    )
  {

    int radius=2;
    __shared__ float s_vx[BLOCK_SIZE+4][BLOCK_SIZE+4];
    __shared__ float s_vz[BLOCK_SIZE+4][BLOCK_SIZE+4];

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int s_tx=tx+radius;  // Thread's x-index into corresponding shared memory tile
    int s_ty=ty+radius;  // Thread's y-index into corresponding shared memory tile

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;
    float dvx_dz,dvz_dx;
    float mu_half_x_half_z;

    int ip;


     if(iz>=1&&iz<ntz-2&&ix>=1&&ix<ntx-2)
     {
    ip=ntx*iz+ix;
    if(tx<radius)
    {
       s_vz[s_ty][tx+BLOCK_SIZE+radius]=vz[ip+BLOCK_SIZE];
       s_vz[s_ty][tx]=vz[ip-radius];
    }
    
    s_vz[s_ty][s_tx]=vz[ip];


    if(ty<radius)
    {
       s_vx[ty+BLOCK_SIZE+radius][s_tx]=vx[ip+BLOCK_SIZE*ntx];
       s_vx[ty][s_tx]=vx[ip-radius*ntx];
    }

    s_vx[s_ty][s_tx]=vx[ip];

    __syncthreads();
     
        dvz_dx=(c[0]*(s_vz[s_ty][s_tx+1]-s_vz[s_ty][s_tx])+
                c[1]*(s_vz[s_ty][s_tx+2]-s_vz[s_ty][s_tx-1]))*one_over_dx;
        dvx_dz=(c[0]*(s_vx[s_ty+1][s_tx]-s_vx[s_ty][s_tx])+
                c[1]*(s_vx[s_ty+2][s_tx]-s_vx[s_ty-1][s_tx]))*one_over_dz;
              
        phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
        phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;

  
        mu_half_x_half_z=0.25f*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);
 
        sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+phi_vz_x[ip]+
                                      dvx_dz+phi_vx_z[ip])*dt+
                                      sigmaxz[ip];
     }

     __syncthreads();
  }
   

  
  __global__ void fdtd_2d_GPU_kernel_vx(
    float *rho,
    float *a_x_half, float *a_z, 
    float *b_x_half, float *b_z,
    float *vx, float *sigmaxx, float *sigmaxz,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    )
  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxx_dx,dsigmaxz_dz;
    float one_over_rho_half_x;

    if(iz>=2&&iz<ntz-1&&ix>=1&&ix<ntx-2)
    {   
       ip=(iz*ntx+ix);
       dsigmaxx_dx=(c[0]*(sigmaxx[ip+1]-sigmaxx[ip])+
                    c[1]*(sigmaxx[ip+2]-sigmaxx[ip-1]))*one_over_dx;
       dsigmaxz_dz=(c[0]*(sigmaxz[ip]-sigmaxz[ip-ntx])+
                    c[1]*(sigmaxz[ip+ntx]-sigmaxz[ip-2*ntx]))*one_over_dz;

       one_over_rho_half_x=1.0f/(0.5f*(rho[ip]+rho[ip+1]));

       vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx
                                     +dsigmaxz_dz)
                                     +vx[ip];
    }

     __syncthreads();
  }



  __global__ void fdtd_2d_GPU_kernel_vz(
    float *rho,
    float *vz, float *sigmaxz, float *sigmazz,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    )
  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxz_dx,dsigmazz_dz;

    float one_over_rho_half_z;

    int ip;


    if(iz>=1&&iz<ntz-2&&ix>=2&&ix<ntx-1)
    {
       ip=iz*ntx+ix;
       dsigmaxz_dx=(c[0]*(sigmaxz[ip]-sigmaxz[ip-1])+
                    c[1]*(sigmaxz[ip+1]-sigmaxz[ip-2]))*one_over_dx;
       dsigmazz_dz=(c[0]*(sigmazz[ip+ntx]-sigmazz[ip])+
                    c[1]*(sigmazz[ip+2*ntx]-sigmazz[ip-ntx]))*one_over_dz;   

       one_over_rho_half_z=1.0f/(0.5f*(rho[ip]+rho[ip+ntx]));
                 
       vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx
                                     +dsigmazz_dz)
                                     +vz[ip];
    }
    __syncthreads();

  }


  __global__ void fdtd_2d_GPU_kernel_sigmaxx_sigmazz(
    float *rick, 
    float *lambda, float *lambda_plus_two_mu,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt, int s_ix, int s_iz, int it
    )
  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dvx_dx,dvz_dz;
    int ip;


    if(iz>=2&&iz<ntz-1&&ix>=2&&ix<ntx-1)
    {
       ip=ntx*iz+ix;
       dvx_dx=(c[0]*(vx[ip]-vx[ip-1])+
               c[1]*(vx[ip+1]-vx[ip-2]))*one_over_dx;
       dvz_dz=(c[0]*(vz[ip]-vz[ip-ntx])+
               c[1]*(vz[ip+ntx]-vz[ip-2*ntx]))*one_over_dz;

       sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx)+
                    lambda[ip]*(dvz_dz))*dt+
                    sigmaxx[ip];

       sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz)+
                    lambda[ip]*(dvx_dx))*dt+
                    sigmazz[ip];

       if(iz==s_iz&&ix==s_ix)
       {
          sigmaxx[ip]=sigmaxx[ip]+rick[it];
          sigmazz[ip]=sigmazz[ip]+rick[it];
       }

    }
    __syncthreads();

  }

  __global__ void fdtd_2d_GPU_kernel_sigmaxz(
    float *mu,
    float *vx, float *vz, float *sigmaxz,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    )
  {
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;
    float dvx_dz,dvz_dx;
    float mu_half_x_half_z;

    int ip;

    if(iz>=1&&iz<ntz-2&&ix>=1&&ix<ntx-2)
    {
       ip=ntx*iz+ix;
       dvz_dx=(c[0]*(vz[ip+1]-vz[ip])+
               c[1]*(vz[ip+2]-vz[ip-1]))*one_over_dx;
       dvx_dz=(c[0]*(vx[ip+ntx]-vx[ip])+
               c[1]*(vx[ip+2*ntx]-vx[ip-ntx]))*one_over_dz;

       mu_half_x_half_z=0.25f*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);
 
       sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+
                                     dvx_dz)*dt+
                                     sigmaxz[ip];
    }

    __syncthreads();
  }

/*==========================================================

  This subroutine is used for calculating wave field in 2D.
   
===========================================================*/

  void fdtd_2d_GPU_backward(int ntx, int ntz, int ntp, int nx, int nz,
                    int pml, float dx, float dz,
                    float *rick, int itmax, float dt,
                    int is, int s_ix, int s_iz, float *rho,
                    float *lambda, float *mu, float *lambda_plus_two_mu,
                    float *k_x, float *k_x_half,
                    float *k_z, float *k_z_half,
                    float *a_x, float *a_x_half,
                    float *a_z, float *a_z_half,
                    float *b_x, float *b_x_half,
                    float *b_z, float *b_z_half,
                    float *seismogram_vx_rms, float *seismogram_vz_rms,
                    float *vx_borders_up, float *vx_borders_bottom,
                    float *vx_borders_left, float *vx_borders_right,
                    float *vz_borders_up, float *vz_borders_bottom,
                    float *vz_borders_left, float *vz_borders_right,
                    float *image_lambda, float *image_mu,
                    float *image_sources, float *image_receivers)
  {
    int it,ip;
  
    float *vx,*vz;
    float *sigmaxx,*sigmaxz,*sigmazz;
//    float vx_temp[ntp],vz_temp[ntp],vx_vz_temp[ntp];
  
    float *phi_vx_x,*phi_vx_z,*phi_vz_z,*phi_vz_x;
  
    float *phi_sigmaxx_x,*phi_sigmaxz_z;
    float *phi_sigmaxz_x,*phi_sigmazz_z;

    float *phi_sigmaxx_z,*phi_sigmazz_x;

//    char filename[30];
    FILE *fp;

    // vectors for the devices

    float *d_rick;
    float *d_lambda, *d_mu, *d_rho;
    float *d_lambda_plus_two_mu;

//    float *d_k_x, *d_k_x_half;
//    float *d_k_z, *d_k_z_half;
    float *d_a_x, *d_a_x_half;
    float *d_a_z, *d_a_z_half;
    float *d_b_x, *d_b_x_half;
    float *d_b_z, *d_b_z_half;

    float *d_vx,*d_vz;
    float *d_sigmaxx,*d_sigmaxz,*d_sigmazz;

//  Wavefields of the constructed by using the storage of the borders...
    float *d_vx_inv,*d_vz_inv;
    float *d_sigmaxx_inv,*d_sigmaxz_inv,*d_sigmazz_inv;  
  
    float *d_phi_vx_x,*d_phi_vx_z,*d_phi_vz_z,*d_phi_vz_x;
  
    float *d_phi_sigmaxx_x,*d_phi_sigmaxz_z;
    float *d_phi_sigmaxz_x,*d_phi_sigmazz_z;
    float *d_phi_sigmaxx_z,*d_phi_sigmazz_x;

    size_t size_model=sizeof(float)*ntp;

// =======================================================
    float *d_seismogram_vx_rms,*d_seismogram_vz_rms;
    float *d_vx_borders_up,*d_vx_borders_bottom;
    float *d_vx_borders_left,*d_vx_borders_right;

    float *d_vz_borders_up,*d_vz_borders_bottom;
    float *d_vz_borders_left,*d_vz_borders_right;
    
    float *d_image_lambda,*d_image_mu;
    float *d_image_sources,*d_image_receivers;

//    int iz,ix;
    int receiver_z=pml+5;

    // allocate the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,...
    vx=(float*)malloc(sizeof(float)*ntp); 
    vz=(float*)malloc(sizeof(float)*ntp); 
    sigmaxx=(float*)malloc(sizeof(float)*ntp);
    sigmazz=(float*)malloc(sizeof(float)*ntp);
    sigmaxz=(float*)malloc(sizeof(float)*ntp);

    // allocate the memory of phi_vx_x...
    phi_vx_x      = (float*)malloc(sizeof(float)*ntp);
    phi_vz_z      = (float*)malloc(sizeof(float)*ntp);
    phi_vx_z      = (float*)malloc(sizeof(float)*ntp);
    phi_vz_x      = (float*)malloc(sizeof(float)*ntp); 
    
    // allocate the memory of phi_sigmaxx_x...
    phi_sigmaxx_x=(float*)malloc(sizeof(float)*ntp);
    phi_sigmaxz_z=(float*)malloc(sizeof(float)*ntp);
    phi_sigmaxz_x=(float*)malloc(sizeof(float)*ntp);
    phi_sigmazz_z=(float*)malloc(sizeof(float)*ntp);

    phi_sigmaxx_z=(float*)malloc(sizeof(float)*ntp);
    phi_sigmazz_x=(float*)malloc(sizeof(float)*ntp);

    // allocate the memory for the device
    cudaMalloc((void**)&d_seismogram_vx_rms,sizeof(float)*itmax*(nx));
    cudaMalloc((void**)&d_seismogram_vz_rms,sizeof(float)*itmax*(nx));

    cudaMalloc((void**)&d_vx_borders_up,sizeof(float)*itmax*(nx+1));
    cudaMalloc((void**)&d_vx_borders_bottom,sizeof(float)*itmax*(nx+1));
    cudaMalloc((void**)&d_vx_borders_left,sizeof(float)*itmax*(nz-2));
    cudaMalloc((void**)&d_vx_borders_right,sizeof(float)*itmax*(nz-2));

    cudaMalloc((void**)&d_vz_borders_up,sizeof(float)*itmax*(nx));
    cudaMalloc((void**)&d_vz_borders_bottom,sizeof(float)*itmax*(nx));
    cudaMalloc((void**)&d_vz_borders_left,sizeof(float)*itmax*(nz-1));
    cudaMalloc((void**)&d_vz_borders_right,sizeof(float)*itmax*(nz-1));

    cudaMalloc((void**)&d_rick,sizeof(float)*itmax);        // ricker wave 

    cudaMalloc((void**)&d_lambda,size_model);
    cudaMalloc((void**)&d_mu,size_model);
    cudaMalloc((void**)&d_rho,size_model);
    cudaMalloc((void**)&d_lambda_plus_two_mu,size_model);   // model 
     

    cudaMalloc((void**)&d_a_x,sizeof(float)*ntx);
    cudaMalloc((void**)&d_a_x_half,sizeof(float)*ntx);
    cudaMalloc((void**)&d_a_z,sizeof(float)*ntz);
    cudaMalloc((void**)&d_a_z_half,sizeof(float)*ntz);

    cudaMalloc((void**)&d_b_x,sizeof(float)*ntx);
    cudaMalloc((void**)&d_b_x_half,sizeof(float)*ntx);
    cudaMalloc((void**)&d_b_z,sizeof(float)*ntz);
    cudaMalloc((void**)&d_b_z_half,sizeof(float)*ntz);      // atten parameters



    cudaMalloc((void**)&d_image_lambda,size_model);
    cudaMalloc((void**)&d_image_mu,size_model);
    cudaMalloc((void**)&d_image_sources,size_model);
    cudaMalloc((void**)&d_image_receivers,size_model);

    cudaMalloc((void**)&d_vx,size_model);
    cudaMalloc((void**)&d_vz,size_model);
    cudaMalloc((void**)&d_sigmaxx,size_model);
    cudaMalloc((void**)&d_sigmazz,size_model);
    cudaMalloc((void**)&d_sigmaxz,size_model);              // wavefields 


    cudaMalloc((void**)&d_vx_inv,size_model);
    cudaMalloc((void**)&d_vz_inv,size_model);
    cudaMalloc((void**)&d_sigmaxx_inv,size_model);
    cudaMalloc((void**)&d_sigmazz_inv,size_model);
    cudaMalloc((void**)&d_sigmaxz_inv,size_model);          // constructed wavefields 


    cudaMalloc((void**)&d_phi_vx_x,size_model);
    cudaMalloc((void**)&d_phi_vz_z,size_model);
    cudaMalloc((void**)&d_phi_vx_z,size_model);
    cudaMalloc((void**)&d_phi_vz_x,size_model);

    cudaMalloc((void**)&d_phi_sigmaxx_x,size_model);
    cudaMalloc((void**)&d_phi_sigmaxz_z,size_model);
    cudaMalloc((void**)&d_phi_sigmaxz_x,size_model);
    cudaMalloc((void**)&d_phi_sigmazz_z,size_model);

    cudaMalloc((void**)&d_phi_sigmaxx_z,size_model);
    cudaMalloc((void**)&d_phi_sigmazz_x,size_model);
    

// =============================================================================

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid((ntx+dimBlock.x-1)/dimBlock.x,(ntz+dimBlock.y-1)/dimBlock.y);

//-----------------------------------------------------------------------//
//=======================================================================//
//-----------------------------------------------------------------------//
  
    fp=fopen("./output/wavefield_itmax.dat","rb");
    fread(&vx[0],sizeof(float),ntp,fp);
    fread(&vz[0],sizeof(float),ntp,fp);

    fread(&sigmaxx[0],sizeof(float),ntp,fp);
    fread(&sigmazz[0],sizeof(float),ntp,fp);
    fread(&sigmaxz[0],sizeof(float),ntp,fp);
    fclose(fp);


    // Copy the vectors from the host to the device

    cudaMemcpy(d_seismogram_vx_rms,seismogram_vx_rms,
               sizeof(float)*(nx)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_seismogram_vz_rms,seismogram_vz_rms,
               sizeof(float)*(nx)*itmax,cudaMemcpyHostToDevice);

    cudaMemcpy(d_vx_borders_up,vx_borders_up,
               sizeof(float)*(nx+1)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx_borders_bottom,vx_borders_bottom,
               sizeof(float)*(nx+1)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx_borders_left,vx_borders_left,
               sizeof(float)*(nz-2)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx_borders_right,vx_borders_right,
               sizeof(float)*(nz-2)*itmax,cudaMemcpyHostToDevice);

    cudaMemcpy(d_vz_borders_up,vz_borders_up,
               sizeof(float)*(nx)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz_borders_bottom,vz_borders_bottom,
               sizeof(float)*(nx)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz_borders_left,vz_borders_left,
               sizeof(float)*(nz-1)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz_borders_right,vz_borders_right,
               sizeof(float)*(nz-1)*itmax,cudaMemcpyHostToDevice);


    cudaMemcpy(d_rick,rick,sizeof(float)*itmax,cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda,lambda,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu,mu,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda_plus_two_mu,lambda_plus_two_mu,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho,rho,size_model,cudaMemcpyHostToDevice);

    cudaMemcpy(d_a_x,a_x,sizeof(float)*ntx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_x_half,a_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_z,a_z,sizeof(float)*ntz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_z_half,a_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice);

    cudaMemcpy(d_b_x,b_x,sizeof(float)*ntx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_x_half,b_x_half,sizeof(float)*ntx,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_z,b_z,sizeof(float)*ntz,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_z_half,b_z_half,sizeof(float)*ntz,cudaMemcpyHostToDevice);


    cudaMemcpy(d_image_lambda,image_lambda,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_image_mu,image_mu,size_model,cudaMemcpyHostToDevice);

    cudaMemcpy(d_image_sources,image_sources,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_image_receivers,image_receivers,size_model,cudaMemcpyHostToDevice);

    cudaMemcpy(d_vx_inv,vx,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz_inv,vz,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmaxx_inv,sigmaxx,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmazz_inv,sigmazz,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmaxz_inv,sigmaxz,size_model,cudaMemcpyHostToDevice);

    // Initialize the fields........................
    
    for(ip=0;ip<ntp;ip++)
    {
        vx[ip]=0.0f;
        vz[ip]=0.0f;
           	
        sigmaxx[ip]=0.0f;
        sigmazz[ip]=0.0f;
        sigmaxz[ip]=0.0f;
          
        phi_vx_x[ip]=0.0f;
        phi_vz_z[ip]=0.0f;
        phi_vx_z[ip]=0.0f;
        phi_vz_x[ip]=0.0f;

        phi_sigmaxx_x[ip]=0.0f;
        phi_sigmaxz_z[ip]=0.0f;
    
        phi_sigmaxz_x[ip]=0.0f;
        phi_sigmazz_z[ip]=0.0f;

        phi_sigmaxx_z[ip]=0.0f;
        phi_sigmazz_x[ip]=0.0f;

//        image_lambda[ip]=0.0;
//        image_mu[ip]=0.0;
//        image_sources[ip]=0.0;
//        image_receivers[ip]=0.0;

    }

    cudaMemcpy(d_vx,vx,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz,vz,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmaxx,sigmaxx,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmazz,sigmazz,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmaxz,sigmaxz,size_model,cudaMemcpyHostToDevice);

    cudaMemcpy(d_phi_vx_x,phi_vx_x,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_vz_z,phi_vz_z,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_vx_z,phi_vx_z,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_vz_x,phi_vz_x,size_model,cudaMemcpyHostToDevice);

    cudaMemcpy(d_phi_sigmaxx_x,phi_sigmaxx_x,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_sigmaxz_z,phi_sigmaxz_z,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_sigmaxz_x,phi_sigmaxz_x,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_sigmazz_z,phi_sigmazz_z,size_model,cudaMemcpyHostToDevice);

    cudaMemcpy(d_phi_sigmaxx_z,phi_sigmaxx_z,size_model,cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi_sigmazz_x,phi_sigmazz_x,size_model,cudaMemcpyHostToDevice);

//==============================================================================
//  THIS SECTION IS USED TO CONSTRUCT THE FORWARD WAVEFIELDS...           
//==============================================================================

    for(it=itmax-2;it>=0;it--)
    {

        fdtd_2d_GPU_kernel_sigmaxx_sigmazz_backward<<<dimGrid,dimBlock>>>
        (
           d_rick, 
           d_lambda, d_lambda_plus_two_mu,
           d_vx_inv, d_vz_inv, d_sigmaxx_inv, d_sigmazz_inv,
           ntp, ntx, ntz, pml, dx, dz, -dt,
           s_ix, s_iz, it
        );

        fdtd_2d_GPU_kernel_sigmaxz_backward<<<dimGrid,dimBlock>>>
        (
           d_mu,
           d_vx_inv, d_vz_inv, d_sigmaxz_inv,
           ntp, ntx, ntz, pml, dx, dz, -dt
        );

        fdtd_2d_GPU_kernel_vx_backward<<<dimGrid,dimBlock>>>
        (
           d_rho, 
           d_vx_inv, d_sigmaxx_inv, d_sigmaxz_inv, 
           ntp, ntx, ntz, pml, dx, dz, -dt
        );

        fdtd_2d_GPU_kernel_vz_backward<<<dimGrid,dimBlock>>>
        (
           d_rho,
           d_vz_inv, d_sigmaxz_inv, d_sigmazz_inv, 
           ntp, ntx, ntz, pml, dx, dz, -dt
        );

        fdtd_2d_GPU_kernel_borders_backward<<<dimGrid,dimBlock>>>
        (
           d_vx_inv,
           d_vx_borders_up, d_vx_borders_bottom,
           d_vx_borders_left, d_vx_borders_right,
           d_vz_inv,
           d_vz_borders_up, d_vz_borders_bottom,
           d_vz_borders_left, d_vz_borders_right,
           ntp, ntx, ntz, pml, it
        );

        ///////////////////////////////////////////////////////////////////////
    
        fdtd_cpml_2d_GPU_kernel_vx_backward<<<dimGrid,dimBlock>>>
        (
           d_rho, d_a_x_half, d_a_z,
           d_b_x_half, d_b_z,
           d_vx, d_sigmaxx, d_sigmaxz,
           d_phi_sigmaxx_x, d_phi_sigmaxz_z,
           ntp, ntx, ntz, -dx, -dz, dt,
           d_seismogram_vx_rms, it, pml,receiver_z
        );

        fdtd_cpml_2d_GPU_kernel_vz_backward<<<dimGrid,dimBlock>>>
        (
           d_rho,
           d_a_x, d_a_z_half,
           d_b_x, d_b_z_half,
           d_vz, d_sigmaxz, d_sigmazz,
           d_phi_sigmaxz_x, d_phi_sigmazz_z,
           ntp, ntx, ntz, -dx, -dz, dt,
           d_seismogram_vz_rms, it, pml,receiver_z
        );

        fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_backward<<<dimGrid,dimBlock>>>
        ( 
           d_lambda, d_lambda_plus_two_mu,
           d_a_x,d_a_z,d_b_x,d_b_z,
           d_vx, d_vz, d_sigmaxx, d_sigmazz,
           d_phi_vx_x,d_phi_vz_z,
           ntp, ntx, ntz, -dx, -dz, dt
        );

        fdtd_cpml_2d_GPU_kernel_sigmaxz_backward<<<dimGrid,dimBlock>>>
        (
           d_mu,
           d_a_x_half, d_a_z_half,
           d_b_x_half, d_b_z_half,
           d_vx, d_vz, d_sigmaxz,
           d_phi_vx_z, d_phi_vz_x,
           ntp, ntx, ntz, -dx, -dz, dt
        );

        sum_image_GPU_kernel_lambda<<<dimGrid,dimBlock>>>
        (
             d_vx_inv,d_vz_inv,d_vx,d_vz,d_sigmaxx,d_sigmazz,
             d_image_lambda,d_image_sources,d_image_receivers,
             ntx,ntz,pml,dx,dz
        );

        sum_image_GPU_kernel_mu<<<dimGrid,dimBlock>>>
        (
          d_vx_inv,d_vz_inv,
          d_sigmaxx,d_sigmazz,d_sigmaxz,
          d_image_mu,ntx,ntz,pml,dx,dz
        );
/*
        if(it%50==0)
        {
          cudaMemcpy(vx,d_vx_inv,sizeof(float)*ntp,cudaMemcpyDeviceToHost);

          sprintf(filename,"./output/%dvz_inv.dat",it);     
          fp=fopen(filename,"wb");
          fwrite(&vz[0],sizeof(float),ntp,fp);
          fclose(fp);
        }
*/

//        if(it%10==0)

//        {

/*
           sprintf(filename,"./output/%dvx_backward.dat",it);
           fp=fopen(filename,"wb");
           fwrite(&vx_temp[0],sizeof(float),ntp,fp);
           fclose(fp);
*/


    }

     
    cudaMemcpy(image_lambda,d_image_lambda,sizeof(float)*ntp,cudaMemcpyDeviceToHost);
    cudaMemcpy(image_mu,d_image_mu,sizeof(float)*ntp,cudaMemcpyDeviceToHost);

    cudaMemcpy(image_sources,d_image_sources,sizeof(float)*ntp,cudaMemcpyDeviceToHost);
    cudaMemcpy(image_receivers,d_image_receivers,sizeof(float)*ntp,cudaMemcpyDeviceToHost);

    //free the memory of Vx,Vy,Vz,Sigmaxx,Sigmayy,Sigmazz...  
    free(vx);
    free(vz);
    free(sigmaxx);
    free(sigmazz);
    free(sigmaxz);
    
   //free the memory of Phi_vx_x....  
    free(phi_vx_x);
    free(phi_vz_z);
    free(phi_vx_z);
    free(phi_vz_x);
    
    //free the memory of Phi_vx_x....  
    free(phi_sigmaxx_x);
    free(phi_sigmaxz_z);
    free(phi_sigmaxz_x);
    free(phi_sigmazz_z);
    free(phi_sigmaxx_z);
    free(phi_sigmazz_x);
    
    //free the memory of DEVICE
    cudaFree(d_seismogram_vx_rms);
    cudaFree(d_seismogram_vz_rms);

    cudaFree(d_vx_borders_up);
    cudaFree(d_vx_borders_bottom);
    cudaFree(d_vx_borders_left);
    cudaFree(d_vx_borders_right);

    cudaFree(d_vz_borders_up);
    cudaFree(d_vz_borders_bottom);
    cudaFree(d_vz_borders_left);
    cudaFree(d_vz_borders_right);

    cudaFree(d_rick);
    cudaFree(d_lambda);
    cudaFree(d_mu);
    cudaFree(d_lambda_plus_two_mu);
    cudaFree(d_rho);
    
    cudaFree(d_a_x);
    cudaFree(d_a_x_half);
    cudaFree(d_a_z);
    cudaFree(d_a_z_half);

    cudaFree(d_b_x);
    cudaFree(d_b_x_half);
    cudaFree(d_b_z);
    cudaFree(d_b_z_half);

    cudaFree(d_image_lambda);
    cudaFree(d_image_mu);
    cudaFree(d_image_sources);
    cudaFree(d_image_receivers);

    cudaFree(d_vx);
    cudaFree(d_vz);
    cudaFree(d_sigmaxx);
    cudaFree(d_sigmazz);
    cudaFree(d_sigmaxz);

    cudaFree(d_vx_inv);
    cudaFree(d_vz_inv);
    cudaFree(d_sigmaxx_inv);
    cudaFree(d_sigmazz_inv);
    cudaFree(d_sigmaxz_inv);
    
    cudaFree(d_phi_vx_x);
    cudaFree(d_phi_vz_z);
    cudaFree(d_phi_vx_z);
    cudaFree(d_phi_vz_x);
      
    cudaFree(d_phi_sigmaxx_x);
    cudaFree(d_phi_sigmaxz_z);
    cudaFree(d_phi_sigmaxz_x);
    cudaFree(d_phi_sigmazz_z);

    cudaFree(d_phi_sigmaxx_z);
    cudaFree(d_phi_sigmazz_x);

  }


  __global__ void fdtd_2d_GPU_kernel_sigmaxx_sigmazz_backward(
    float *rick, 
    float *lambda, float *lambda_plus_two_mu,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    int ntp, int ntx, int ntz, int pml,
    float dx, float dz, float dt, int s_ix, int s_iz, int it
    )

  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dvx_dx,dvz_dz;
    int ip;


    if(iz>=pml+1&&iz<ntz-pml-1&&ix>=pml+1&&ix<ntx-pml-1)
    {
       ip=ntx*iz+ix;
       dvx_dx=(c[0]*(vx[ip]-vx[ip-1])+
               c[1]*(vx[ip+1]-vx[ip-2]))*one_over_dx;
       dvz_dz=(c[0]*(vz[ip]-vz[ip-ntx])+
               c[1]*(vz[ip+ntx]-vz[ip-2*ntx]))*one_over_dz;

       sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx)+
                    lambda[ip]*(dvz_dz))*dt+
                    sigmaxx[ip];

       sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz)+
                    lambda[ip]*(dvx_dx))*dt+
                    sigmazz[ip];

       if(iz==s_iz&&ix==s_ix)
       {
          sigmaxx[ip]=sigmaxx[ip]-rick[it+1];
          sigmazz[ip]=sigmazz[ip]-rick[it+1];
       }

    }

    if(iz==pml&&ix>=pml&&ix<=ntx-pml-1||
       iz==ntz-pml-1&&ix>=pml&&ix<=ntx-pml-1||
       ix==pml&&iz>=pml+1&&iz<=ntz-pml-2||
       ix==ntx-pml-1&&iz>=pml+1&&iz<=ntz-pml-2)
    {
       ip=ntx*iz+ix;

       dvx_dx=(vx[ip]-vx[ip-1])*one_over_dx;
       dvz_dz=(vz[ip]-vz[ip-ntx])*one_over_dz;

       sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx)+
                    lambda[ip]*(dvz_dz))*dt+
                    sigmaxx[ip];

       sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz)+
                    lambda[ip]*(dvx_dx))*dt+
                    sigmazz[ip];
    }
      
    __syncthreads();

  }

  __global__ void fdtd_2d_GPU_kernel_sigmaxz_backward(
    float *mu,
    float *vx, float *vz, float *sigmaxz,
    int ntp, int ntx, int ntz, int pml,
    float dx, float dz, float dt
    )
  {
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;
    float dvx_dz,dvz_dx;
    float mu_half_x_half_z;

    int ip;

    if(iz>=pml+1&&iz<ntz-pml-2&&ix>=pml+1&&ix<ntx-pml-2)
    {
       ip=ntx*iz+ix;

       dvz_dx=(c[0]*(vz[ip+1]-vz[ip])+
               c[1]*(vz[ip+2]-vz[ip-1]))*one_over_dx;
       dvx_dz=(c[0]*(vx[ip+ntx]-vx[ip])+
               c[1]*(vx[ip+2*ntx]-vx[ip-ntx]))*one_over_dz;

       mu_half_x_half_z=0.25f*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);
 
       sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+
                                     dvx_dz)*dt+
                                     sigmaxz[ip];
    }

    if(iz==pml&&ix>=pml&&ix<=ntx-pml-2||
       iz==ntz-pml-2&&ix>=pml&&ix<=ntx-pml-2||
       ix==pml&&iz>=pml+1&&iz<=ntz-pml-3||
       ix==ntx-pml-2&&iz>=pml+1&&iz<=ntz-pml-3)

    {
       ip=ntx*iz+ix;

       dvz_dx=(vz[ip+1]-vz[ip])*one_over_dx;
       dvx_dz=(vx[ip+ntx]-vx[ip])*one_over_dz;

       mu_half_x_half_z=0.25f*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);
 
       sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+
                                     dvx_dz)*dt+
                                     sigmaxz[ip];
    }

    __syncthreads();
  }


  __global__ void fdtd_2d_GPU_kernel_vx_backward(
    float *rho,
    float *vx, float *sigmaxx, float *sigmaxz,
    int ntp, int ntx, int ntz, int pml,
    float dx, float dz, float dt
    )
  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxx_dx,dsigmaxz_dz;
    float one_over_rho_half_x;

    if(iz>=pml+2&&iz<=ntz-pml-3&&ix>=pml+1&&ix<=ntx-pml-3)
    {   
       ip=(iz*ntx+ix);
       dsigmaxx_dx=(c[0]*(sigmaxx[ip+1]-sigmaxx[ip])+
                    c[1]*(sigmaxx[ip+2]-sigmaxx[ip-1]))*one_over_dx;
       dsigmaxz_dz=(c[0]*(sigmaxz[ip]-sigmaxz[ip-ntx])+
                    c[1]*(sigmaxz[ip+ntx]-sigmaxz[ip-2*ntx]))*one_over_dz;

       one_over_rho_half_x=1.0f/(0.5f*(rho[ip]+rho[ip+1]));

       vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx
                                     +dsigmaxz_dz)
                                     +vx[ip];
    }

    if(iz==pml+1&&ix>=pml&&ix<=ntx-pml-2||
       iz==ntz-pml-2&&ix>=pml&&ix<=ntx-pml-2||
       ix==pml&&iz>=pml+2&&iz<=ntz-pml-3||
       ix==ntx-pml-2&&iz>=pml+2&&iz<=ntz-pml-3)
    {

       ip=(iz*ntx+ix);
       dsigmaxx_dx=(sigmaxx[ip+1]-sigmaxx[ip])*one_over_dx;
       dsigmaxz_dz=(sigmaxz[ip]-sigmaxz[ip-ntx])*one_over_dz;

       one_over_rho_half_x=1.0f/(0.5f*(rho[ip]+rho[ip+1]));

       vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx
                                     +dsigmaxz_dz)
                                     +vx[ip];
    }
     __syncthreads();
  }

  __global__ void fdtd_2d_GPU_kernel_vz_backward(
    float *rho,
    float *vz, float *sigmaxz, float *sigmazz,
    int ntp, int ntx, int ntz, int pml,
    float dx, float dz, float dt
    )
  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxz_dx,dsigmazz_dz;

    float one_over_rho_half_z;

    int ip;


    if(iz>=pml+1&&iz<=ntz-pml-3&&ix>=pml+2&&ix<=ntx-pml-3)
    {
       ip=iz*ntx+ix;
       dsigmaxz_dx=(c[0]*(sigmaxz[ip]-sigmaxz[ip-1])+
                    c[1]*(sigmaxz[ip+1]-sigmaxz[ip-2]))*one_over_dx;
       dsigmazz_dz=(c[0]*(sigmazz[ip+ntx]-sigmazz[ip])+
                    c[1]*(sigmazz[ip+2*ntx]-sigmazz[ip-ntx]))*one_over_dz;   

       one_over_rho_half_z=1.0f/(0.5f*(rho[ip]+rho[ip+ntx]));
                 
       vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx
                                     +dsigmazz_dz)
                                     +vz[ip];
    }

    if(iz==pml&&ix>=pml+1&&ix<=ntx-pml-2||
       iz==ntz-pml-2&&ix>=pml+1&&ix<=ntx-pml-2||
       ix==pml+1&&iz>=pml+1&&iz<=ntz-pml-3||
       ix==ntx-pml-2&&iz>=pml+1&&iz<=ntz-pml-3)
    {

       ip=iz*ntx+ix;
       dsigmaxz_dx=(sigmaxz[ip]-sigmaxz[ip-1])*one_over_dx;
       dsigmazz_dz=(sigmazz[ip+ntx]-sigmazz[ip])*one_over_dz;   

       one_over_rho_half_z=1.0f/(0.5f*(rho[ip]+rho[ip+ntx]));
                 
       vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx
                                     +dsigmazz_dz)
                                     +vz[ip];
    }

    __syncthreads();

  }

  __global__ void fdtd_2d_GPU_kernel_borders_backward
  (
   float *vx,
   float *vx_borders_up, float *vx_borders_bottom,
   float *vx_borders_left, float *vx_borders_right,
   float *vz,
   float *vz_borders_up, float *vz_borders_bottom,
   float *vz_borders_left, float *vz_borders_right,
   int ntp, int ntx, int ntz, int pml, int it
  )
  {


    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip=iz*ntx+ix;
    int nx=ntx-2*pml;
    int nz=ntz-2*pml;


      if(ix>=pml-1&&ix<=ntx-pml-1&&iz==pml)
       {
          vx[ip]=vx_borders_up[it*(nx+1)+ix-pml+1];
       }
       if(ix>=pml-1&&ix<=ntx-pml-1&&iz==ntz-pml-1)
       {
          vx[ip]=vx_borders_bottom[it*(nx+1)+ix-pml+1];
       }


       if(iz>=pml+1&&iz<=ntz-pml-2&&ix==pml-1)
       {
          vx[ip]=vx_borders_left[it*(nz-2)+iz-pml-1];
       }
       if(iz>=pml+1&&iz<=ntz-pml-2&&ix==ntx-pml-1)
       {
          vx[ip]=vx_borders_right[it*(nz-2)+iz-pml-1];
       }


    __syncthreads();


       if(ix>=pml&&ix<=ntx-pml-1&&iz==pml-1)
       {
          vz[ip]=vz_borders_up[it*(nx)+ix-pml];
       }
       if(ix>=pml&&ix<=ntx-pml-1&&iz==ntz-pml-1)
       {
          vz[ip]=vz_borders_bottom[it*(nx)+ix-pml];
       }

       if(iz>=pml&&iz<=ntz-pml-2&&ix==pml)
       {
          vz[ip]=vz_borders_left[it*(nz-1)+iz-pml];
       }
       if(iz>=pml&&iz<=ntz-pml-2&&ix==ntx-pml-1)
       {

          vz[ip]=vz_borders_right[it*(nz-1)+iz-pml];
       }
    __syncthreads();

  }


  __global__ void fdtd_cpml_2d_GPU_kernel_vx_backward(
    float *rho,
    float *a_x_half, float *a_z, 
    float *b_x_half, float *b_z, 
    float *vx, float *sigmaxx, float *sigmaxz,
    float *phi_sigmaxx_x, float *phi_sigmaxz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt,
    float *seismogram_vx_rms, int it, int pml,
    int receiver_z
    )

  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxx_dx,dsigmaxz_dz;
    float one_over_rho_half_x;

    if(iz>=2&&iz<ntz-1&&ix>=1&&ix<ntx-2)
    {
       ip=iz*ntx+ix;
       dsigmaxx_dx=(c[0]*(sigmaxx[ip+1]-sigmaxx[ip])+
                    c[1]*(sigmaxx[ip+2]-sigmaxx[ip-1]))*one_over_dx;
       dsigmaxz_dz=(c[0]*(sigmaxz[ip]-sigmaxz[ip-ntx])+
                    c[1]*(sigmaxz[ip+ntx]-sigmaxz[ip-2*ntx]))*one_over_dz;
                 
       phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
       phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

       one_over_rho_half_x=1.0f/(0.5f*(rho[ip]+rho[ip+1]));

                
       vx[ip]=dt*one_over_rho_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip]
                                     +dsigmaxz_dz+phi_sigmaxz_z[ip])
                                     +vx[ip];


    }
    __syncthreads();


    // Seismogram...   
    if(ix>=pml&&ix<=ntx-pml-1&&iz==receiver_z)
    {
       vx[ip]=seismogram_vx_rms[it*(ntx-2*pml)+ix-pml];
    }
    __syncthreads();


  }


  __global__ void fdtd_cpml_2d_GPU_kernel_vz_backward(
    float *rho,
    float *a_x, float *a_z_half,
    float *b_x, float *b_z_half,
    float *vz, float *sigmaxz, float *sigmazz, 
    float *phi_sigmaxz_x, float *phi_sigmazz_z,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt,
    float *seismogram_vz_rms, int it, int pml,
    int receiver_z
    )
  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxz_dx,dsigmazz_dz;

    float one_over_rho_half_z;

    int ip;

    if(iz>=1&&iz<ntz-2&&ix>=2&&ix<ntx-1)
    {
       ip=iz*ntx+ix;
       dsigmaxz_dx=(c[0]*(sigmaxz[ip]-sigmaxz[ip-1])+
                    c[1]*(sigmaxz[ip+1]-sigmaxz[ip-2]))*one_over_dx;
       dsigmazz_dz=(c[0]*(sigmazz[ip+ntx]-sigmazz[ip])+
                    c[1]*(sigmazz[ip+2*ntx]-sigmazz[ip-ntx]))*one_over_dz;
                 
       phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;
       phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;


       one_over_rho_half_z=1.0f/(0.5f*(rho[ip]+rho[ip+ntx]));
  
               
       vz[ip]=dt*one_over_rho_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip]
                                     +dsigmazz_dz+phi_sigmazz_z[ip])
                                     +vz[ip];

    }
    __syncthreads();

    if(ix>=pml&&ix<=ntx-pml-1&&iz==receiver_z)
    {
       vz[ip]=seismogram_vz_rms[it*(ntx-2*pml)+ix-pml];
    }
    __syncthreads();

  }

  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_backward(
    float *lambda, float *lambda_plus_two_mu,
    float *a_x, float *a_z,
    float *b_x, float *b_z,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    float *phi_vx_x, float *phi_vz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    )

  {
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dvx_dx,dvz_dz;
    int ip;

    if(iz>=2&&iz<ntz-1&&ix>=2&&ix<ntx-1)
    {
       ip=iz*ntx+ix;
       dvx_dx=(c[0]*(vx[ip]-vx[ip-1])+
               c[1]*(vx[ip+1]-vx[ip-2]))*one_over_dx;
       dvz_dz=(c[0]*(vz[ip]-vz[ip-ntx])+
               c[1]*(vz[ip+ntx]-vz[ip-2*ntx]))*one_over_dz;
            
       phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
       phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;

            
       sigmaxx[ip]=(lambda_plus_two_mu[ip]*(dvx_dx+phi_vx_x[ip])+
                    lambda[ip]*(dvz_dz+phi_vz_z[ip]))*dt+
                    sigmaxx[ip];

       sigmazz[ip]=(lambda_plus_two_mu[ip]*(dvz_dz+phi_vz_z[ip])+
                    lambda[ip]*(dvx_dx+phi_vx_x[ip]))*dt+
                    sigmazz[ip];


    }

    __syncthreads();
  }

  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_backward(
    float *mu,
    float *a_x_half, float *a_z_half,
    float *b_x_half, float *b_z_half,
    float *vx, float *vz, float *sigmaxz,
    float *phi_vx_z, float *phi_vz_x,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    )
  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;
    float dvx_dz,dvz_dx;
    float mu_half_x_half_z;

    int ip;

     if(iz>=1&&iz<ntz-2&&ix>=1&&ix<ntx-2)
     {
        ip=iz*ntx+ix;      
        dvz_dx=(c[0]*(vz[ip+1]-vz[ip])+
                c[1]*(vz[ip+2]-vz[ip-1]))*one_over_dx;
        dvx_dz=(c[0]*(vx[ip+ntx]-vx[ip])+
                c[1]*(vx[ip+2*ntx]-vx[ip-ntx]))*one_over_dz;
              
        phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
        phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;

  
        mu_half_x_half_z=0.25f*(mu[ip]+mu[ip+ntx]+mu[ip+1]+mu[ip+ntx+1]);
 
        sigmaxz[ip]=mu_half_x_half_z*(dvz_dx+phi_vz_x[ip]+
                                      dvx_dz+phi_vx_z[ip])*dt+
                                      sigmaxz[ip];
     }

     __syncthreads();
  }

  __global__ void fdtd_cpml_2d_GPU_kernel_vx_mine_backward(
    float *rho, float *lambda, float *mu, float *lambda_plus_two_mu,
    float *a_x_half, float *a_z, 
    float *b_x_half, float *b_z, 
    float *vx, float *sigmaxx, float *sigmazz, float *sigmaxz,
    float *phi_sigmaxx_x, float *phi_sigmazz_x, float *phi_sigmaxz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt,
    float *seismogram_vx_rms, int it, int pml,
    int receiver_z
    )

  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxx_dx,dsigmazz_dx,dsigmaxz_dz;
    float one_over_rho_half_x;
    float lambda_half_x, mu_half_x,lambda_plus_two_mu_half_x;

    if(iz>=2&&iz<ntz-1&&ix>=1&&ix<ntx-2)
    {
       ip=iz*ntx+ix;
       dsigmaxx_dx=(c[0]*(sigmaxx[ip+1]-sigmaxx[ip])+
                    c[1]*(sigmaxx[ip+2]-sigmaxx[ip-1]))*one_over_dx;
       dsigmazz_dx=(c[0]*(sigmazz[ip+1]-sigmazz[ip])+
                    c[1]*(sigmazz[ip+2]-sigmazz[ip-1]))*one_over_dx;
       dsigmaxz_dz=(c[0]*(sigmaxz[ip]-sigmaxz[ip-ntx])+
                    c[1]*(sigmaxz[ip+ntx]-sigmaxz[ip-2*ntx]))*one_over_dz;
                 
       phi_sigmaxx_x[ip]=b_x_half[ix]*phi_sigmaxx_x[ip]+a_x_half[ix]*dsigmaxx_dx;
       phi_sigmazz_x[ip]=b_x_half[ix]*phi_sigmazz_x[ip]+a_x_half[ix]*dsigmazz_dx;
       phi_sigmaxz_z[ip]=b_z[iz]*phi_sigmaxz_z[ip]+a_z[iz]*dsigmaxz_dz;

       one_over_rho_half_x=1.0f/(0.5f*(rho[ip]+rho[ip+1]));
       lambda_half_x=(lambda[ip]+lambda[ip+1])/2.0f;
       mu_half_x=(mu[ip]+mu[ip+1])/2.0f;
       lambda_plus_two_mu_half_x=(lambda_plus_two_mu[ip]+lambda_plus_two_mu[ip+1])/2.0f;

                
       vx[ip]=dt*one_over_rho_half_x*(lambda_plus_two_mu_half_x*(dsigmaxx_dx+phi_sigmaxx_x[ip])
                                     +lambda_half_x*(dsigmazz_dx+phi_sigmazz_x[ip])
                                     +mu_half_x*(dsigmaxz_dz+phi_sigmaxz_z[ip]))
                                     +vx[ip];

    }
    __syncthreads();


    // Seismogram...   
    if(ix>=pml&&ix<=ntx-pml-1&&iz==receiver_z)
    {
       vx[ip]=seismogram_vx_rms[it*(ntx-2*pml)+ix-pml];
    }
    __syncthreads();

  }


  __global__ void fdtd_cpml_2d_GPU_kernel_vz_mine_backward(
    float *rho, float *lambda, float *mu, float *lambda_plus_two_mu,
    float *a_x, float *a_z_half,
    float *b_x, float *b_z_half,
    float *vz, float *sigmaxx, float *sigmazz, float *sigmaxz, 
    float *phi_sigmaxx_z, float *phi_sigmazz_z, float *phi_sigmaxz_x, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt,
    float *seismogram_vz_rms, int it, int pml,
    int receiver_z
    )

  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dsigmaxx_dz,dsigmazz_dz,dsigmaxz_dx;
    float one_over_rho_half_z;
    float lambda_half_z, mu_half_z,lambda_plus_two_mu_half_z;

    int ip;

    if(iz>=1&&iz<ntz-2&&ix>=2&&ix<ntx-1)
    {
       ip=iz*ntx+ix;

       dsigmaxx_dz=(c[0]*(sigmaxx[ip+ntx]-sigmaxx[ip])+
                    c[1]*(sigmaxx[ip+2*ntx]-sigmaxx[ip-ntx]))*one_over_dz;
       dsigmaxz_dx=(c[0]*(sigmaxz[ip]-sigmaxz[ip-1])+
                    c[1]*(sigmaxz[ip+1]-sigmaxz[ip-2]))*one_over_dx;
       dsigmazz_dz=(c[0]*(sigmazz[ip+ntx]-sigmazz[ip])+
                    c[1]*(sigmazz[ip+2*ntx]-sigmazz[ip-ntx]))*one_over_dz;
                 
       phi_sigmaxz_x[ip]=b_x[ix]*phi_sigmaxz_x[ip]+a_x[ix]*dsigmaxz_dx;
       phi_sigmaxx_z[ip]=b_z_half[iz]*phi_sigmaxx_z[ip]+a_z_half[iz]*dsigmaxx_dz;
       phi_sigmazz_z[ip]=b_z_half[iz]*phi_sigmazz_z[ip]+a_z_half[iz]*dsigmazz_dz;


       one_over_rho_half_z=1.0f/(0.5f*(rho[ip]+rho[ip+ntx]));
       lambda_half_z=(lambda[ip]+lambda[ip+ntx])/2.0f;
       mu_half_z=(mu[ip]+mu[ip+ntx])/2.0f;
       lambda_plus_two_mu_half_z=(lambda_plus_two_mu[ip]+lambda_plus_two_mu[ip+ntx])/2.0f;
               
       vz[ip]=dt*one_over_rho_half_z*(mu_half_z*(dsigmaxz_dx+phi_sigmaxz_x[ip])
                                     +lambda_half_z*(dsigmaxx_dz+phi_sigmaxx_z[ip])
                                     +lambda_plus_two_mu_half_z*(dsigmazz_dz+phi_sigmazz_z[ip]))
                                     +vz[ip];

    }
    __syncthreads();

    if(ix>=pml&&ix<=ntx-pml-1&&iz==receiver_z)
    {
       vz[ip]=seismogram_vz_rms[it*(ntx-2*pml)+ix-pml];
    }
    __syncthreads();

  }
    
  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_mine_backward(
    float *a_x, float *a_z,
    float *b_x, float *b_z,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    float *phi_vx_x, float *phi_vz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    )

  {
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;


    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;

    float dvx_dx,dvz_dz;
    int ip;

    if(iz>=2&&iz<ntz-1&&ix>=2&&ix<ntx-1)
    {
       ip=iz*ntx+ix;
       dvx_dx=(c[0]*(vx[ip]-vx[ip-1])+
               c[1]*(vx[ip+1]-vx[ip-2]))*one_over_dx;
       dvz_dz=(c[0]*(vz[ip]-vz[ip-ntx])+
               c[1]*(vz[ip+ntx]-vz[ip-2*ntx]))*one_over_dz;
            
       phi_vx_x[ip]=b_x[ix]*phi_vx_x[ip]+a_x[ix]*dvx_dx;
       phi_vz_z[ip]=b_z[iz]*phi_vz_z[ip]+a_z[iz]*dvz_dz;

            
       sigmaxx[ip]=(dvx_dx+phi_vx_x[ip])*dt+sigmaxx[ip];

       sigmazz[ip]=(dvz_dz+phi_vz_z[ip])*dt+sigmazz[ip];

    }

    __syncthreads();
  }

  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_mine_backward(
    float *a_x_half, float *a_z_half,
    float *b_x_half, float *b_z_half,
    float *vx, float *vz, float *sigmaxz,
    float *phi_vx_z, float *phi_vz_x,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    )
  {

    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    float c[2]={9.0f/8.0f,-1.0f/24.0f};
    float one_over_dx=1.0f/dx;
    float one_over_dz=1.0f/dx;
    float dvx_dz,dvz_dx;

    int ip;

     if(iz>=1&&iz<ntz-2&&ix>=1&&ix<ntx-2)
     {
        ip=iz*ntx+ix;      
        dvz_dx=(c[0]*(vz[ip+1]-vz[ip])+
                c[1]*(vz[ip+2]-vz[ip-1]))*one_over_dx;
        dvx_dz=(c[0]*(vx[ip+ntx]-vx[ip])+
                c[1]*(vx[ip+2*ntx]-vx[ip-ntx]))*one_over_dz;
              
        phi_vz_x[ip]=b_x_half[ix]*phi_vz_x[ip]+a_x_half[ix]*dvz_dx;
        phi_vx_z[ip]=b_z_half[iz]*phi_vx_z[ip]+a_z_half[iz]*dvx_dz;
 
        sigmaxz[ip]=(dvz_dx+phi_vz_x[ip]+
                     dvx_dz+phi_vx_z[ip])*dt+
                     sigmaxz[ip];
     }

     __syncthreads();
  }


  __global__ void sum_image_GPU_kernel_lambda
  (
    float *vx_inv, float *vz_inv, 
    float *vx, float *vz,
    float *sigmaxx, float *sigmazz, 
    float *image_lambda, float *image_sources, float *image_receivers, 
    int ntx, int ntz, int pml, float dx, float dz
  )

  {
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip=iz*ntx+ix;

    float dvx_dx,dvz_dz;
//    float dvx_dx_b,dvz_dz_b;

    if(iz>=pml+1&&iz<=ntz-pml-1&&ix>=pml+1&&ix<=ntx-pml-1)
    {
       dvx_dx=(vx_inv[ip]-vx_inv[ip-1])/dx;
       dvz_dz=(vz_inv[ip]-vz_inv[ip-ntx])/dz;
//       dvx_dx_b=(vx[ip]-vx[ip-1])/dx;
//       dvz_dz_b=(vz[ip]-vz[ip-ntx])/dz;
//       image[ip]=image[ip]-(dvx_dx+dvz_dz)*(dvx_dx_b+dvz_dz_b);
       image_lambda[ip]=image_lambda[ip]+
                 (sigmaxx[ip]+sigmazz[ip])*(dvx_dx+dvz_dz);
       image_sources[ip]=image_sources[ip]+(dvx_dx+dvz_dz)*(dvx_dx+dvz_dz);
       image_receivers[ip]=image_receivers[ip]+
                 (sigmaxx[ip]+sigmazz[ip])*(sigmaxx[ip]+sigmazz[ip]);        
    }
     __syncthreads();
  }



  __global__ void sum_image_GPU_kernel_mu
  (
    float *vx_inv, float *vz_inv, 
    float *sigmaxx, float *sigmazz, float *sigmaxz,
    float *image, 
    int ntx, int ntz, int pml, float dx, float dz
  )

  {
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip=iz*ntx+ix;

    float dvx_dx,dvz_dz;
    float dvz_dx,dvx_dz;

    if(iz>=pml+1&&iz<=ntz-pml-1&&ix>=pml+1&&ix<=ntx-pml-1)
    {
       dvx_dx=(vx_inv[ip]-vx_inv[ip-1])/dx;
       dvz_dz=(vz_inv[ip]-vz_inv[ip-ntx])/dz;
       dvx_dz=(vx_inv[ip+ntx]-vx_inv[ip])/dz;
       dvz_dx=(vz_inv[ip+1]-vz_inv[ip])/dx;
       image[ip]=image[ip]+(sigmaxz[ip]*(dvx_dz+dvz_dx))-
                 2.0f*(sigmaxx[ip]*dvz_dz+sigmazz[ip]*dvx_dx);
       //image[ip]=image[ip]+
       //          2.0f*(sigmaxx[ip]*dvx_dx+sigmazz[ip]*dvz_dz)+
       //          sigmaxz[ip]*(dvx_dz+dvz_dx);
    }
     __syncthreads();
  }


  __global__ void sum_image_GPU_kernel_sources
  (
    float *vx_inv, float *vz_inv, 
    float *image, 
    int ntx, int ntz, int pml, float dx, float dz
  )

  {
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip=iz*ntx+ix;

    float dvx_dx,dvz_dz;

    if(iz>=pml+1&&iz<=ntz-pml-1&&ix>=pml+1&&ix<=ntx-pml-1)
    {
       dvx_dx=(vx_inv[ip]-vx_inv[ip-1])/dx;
       dvz_dz=(vz_inv[ip]-vz_inv[ip-ntx])/dz;

       image[ip]=image[ip]+(dvx_dx+dvz_dz)*(dvx_dx+dvz_dz);
    }
     __syncthreads();
  }


  __global__ void sum_image_GPU_kernel_receivers
  (
    float *sigmaxx, float *sigmazz, float *image, 
    int ntx, int ntz, int pml, float dx, float dz
  )

  {
    int bx=blockIdx.x;
    int by=blockIdx.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    int iz=by*BLOCK_SIZE+ty;
    int ix=bx*BLOCK_SIZE+tx;

    int ip=iz*ntx+ix;

    if(iz>=pml+1&&iz<=ntz-pml-1&&ix>=pml+1&&ix<=ntx-pml-1)
    {
       image[ip]=image[ip]+
                 (sigmaxx[ip]+sigmazz[ip])*(sigmaxx[ip]+sigmazz[ip]);
    }
     __syncthreads();
  }

  void conjugate_gradient(
       float *Gradient_lambda, float *Gradient_mu, 
       float *vp, float *vs, float *rho,
       float *lambda, float *mu, float *lambda_plus_two_mu,
       float *vp_n, float *vs_n,
       int ntp, int ntz, int ntx, int nz, int nx, int pml, float dz, float dx,
       float *rick, int itmax, float dt,
       int ns, int *s_ix, int *s_iz,
       float *k_x, float *k_x_half, float *k_z, float *k_z_half,
       float *a_x, float *a_x_half, float *a_z, float *a_z_half,
       float *b_x, float *b_x_half, float *b_z, float *b_z_half,
       float *c,
       float *seismogram_vx_syn, float *seismogram_vz_syn,
       float *vx_borders_up, float *vx_borders_bottom,
       float *vx_borders_left, float *vx_borders_right,
       float *vz_borders_up, float *vz_borders_bottom,
       float *vz_borders_left, float *vz_borders_right, 
       int iter, int inv_flag,
       float misfit,
       float *Gradient_vp_pre, float *Gradient_vs_pre,
       float *dn_vp_pre, float *dn_vs_pre)

  {

    float P[nz];
    float *Gradient_vp_all,*Gradient_vp;
    float *Gradient_vs_all,*Gradient_vs;

    float *dn_vp,*dn_vs;

    float *seismogram_vx_obs,*seismogram_vz_obs;
    float *seismogram_vx_rms,*seismogram_vz_rms;


    int np=(ntz-2*pml)*(ntx-2*pml);
    int ip,ipp,iz,ix,is;

    FILE *fp;

    float sum1,sum2,beta;
    float Misfit_pre;
    float *un0_vp,*un0_vs;

    float amp_scale=1.0f;
  
    char filename[230];
    float Misfit_new;

    float sum_gradient;


    Gradient_vp_all=(float*)malloc(sizeof(float)*ntp);
    Gradient_vs_all=(float*)malloc(sizeof(float)*ntp);

    Gradient_vp    =(float*)malloc(sizeof(float)*np);
    Gradient_vs    =(float*)malloc(sizeof(float)*np);

    dn_vp          =(float*)malloc(sizeof(float)*np);
    dn_vs          =(float*)malloc(sizeof(float)*np);


    un0_vp         =(float*)malloc(sizeof(float)*1);
    un0_vs         =(float*)malloc(sizeof(float)*1);

    seismogram_vx_obs  =(float*)malloc(sizeof(float)*nx*itmax);
    seismogram_vz_obs  =(float*)malloc(sizeof(float)*nx*itmax);
    seismogram_vx_rms  =(float*)malloc(sizeof(float)*nx*itmax);
    seismogram_vz_rms  =(float*)malloc(sizeof(float)*nx*itmax);



/*  ---------------------------------------------------------------
    Gradient of P wave velocity...
    -------------------------------------------------------------*/

    for(ip=0;ip<ntp;ip++)
    {
        Gradient_vp_all[ip]=-2.0f*rho[ip]*vp_n[ip]*Gradient_lambda[ip]*amp_scale;
    }

//     smooth_model(Gradient_vp_all, ntp, ntx, ntz, pml);

/*
    for(ip=0;ip<ntp;ip++)
    {
//        Gradient_vs_all[ip]=Gradient_vp_all[ip];
          Gradient_vs_all[ip]=Gradient_vp_all[ip];
//        Gradient_vs_all[ip]=-2.0f*rho[ip]*vs_n[ip]*Gradient_mu[ip]*amp_scale;
    }
*/

    for(iz=pml;iz<=ntz-pml-1;iz++)
    {
        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
           ip=iz*ntx+ix;
           ipp=(iz-pml)*nx+ix-pml;
           Gradient_vp[ipp]=Gradient_vp_all[ip];    // inner gradient...
        }
    }


/*
    for(iz=pml;iz<=ntz-pml-1;iz++)
    {
        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
           ip=iz*ntx+ix;
           ipp=(iz-pml)*nx+ix-pml;
           Gradient_vs[ipp]=Gradient_vs_all[ip];    // inner gradient...
        }
    }
*/
    Preprocess(nz,nx,dx,dz,P);
//    Preprocess_vs(nz,nx,dx,dz,P_vs);

    for(ip=0;ip<np;ip++)
    {
        iz=ip/nx;

        Gradient_vp[ip]=Gradient_vp[ip]*P[iz];
//        Gradient_vs[ip]=Gradient_vs[ip]*P[iz];
    }

    

/*==========================================================
  Substract the anomoly points 
  ==========================================================*/
     sum_gradient=0.0;
     for(ip=0;ip<np;ip++)
     {
        sum_gradient=sum_gradient+abs(Gradient_vp[ip]);
     }
     sum_gradient=sum_gradient/np;

     for(ip=0;ip<np;ip++)
     {
         if(abs(Gradient_vp[ip])>10.0*sum_gradient)
         {
            Gradient_vp[ip]=Gradient_vp[ip]/abs(Gradient_vp[ip])*10.0*sum_gradient;
         }
     }

     if(iter>30)
     {
        for(ip=0;ip<np;ip++)
        {
            if(abs(Gradient_vp[ip])>5.0*sum_gradient)
            {
               Gradient_vp[ip]=Gradient_vp[ip]/abs(Gradient_vp[ip])*5.0*sum_gradient;
            }
        }
     }
         
/*==========================================================
    Applying the conjugate gradient method...
  ==========================================================*/

    if(iter==0)
    {
      for(ip=0;ip<np;ip++)
      {
          dn_vp[ip]=-Gradient_vp[ip];
//          dn_vs[ip]=-Gradient_vs[ip];
      }
    }

    if(iter>=1)
    {
       sum1=0.0f;
       sum2=0.0f;

       for(ip=0;ip<np;ip++)
       {
           sum1=sum1+Gradient_vp[ip]*Gradient_vp[ip];
//           sum1=sum1+Gradient_vs[ip]*Gradient_vs[ip];
           sum2=sum2+Gradient_vp_pre[ip]*Gradient_vp_pre[ip];
//           sum2=sum2+Gradient_vs_pre[ip]*Gradient_vs_pre[ip];
       }

       beta=sum1/sum2;

       for(ip=0;ip<np;ip++)
       {
           dn_vp[ip]=-Gradient_vp[ip]+beta*dn_vp_pre[ip];
//           dn_vs[ip]=-Gradient_vs[ip]+beta*dn_vs_pre[ip];
       }
    }


   
    for(ip=0;ip<np;ip++)
    {
        Gradient_vp_pre[ip]=Gradient_vp[ip];
        dn_vp_pre[ip]=dn_vp[ip];

//        Gradient_vs_pre[ip]=Gradient_vs[ip];
//        dn_vs_pre[ip]=dn_vs[ip];
    }


    sprintf(filename,"./output/%dGradient_vp.dat",iter+1);

    fp=fopen(filename,"wb");
    fwrite(&Gradient_vp[0],sizeof(float),np,fp);
    fclose(fp);


//	---------------------------------------------------------------
//	------------calculate the step --------------------------------
//	---------------------------------------------------------------

//	***************************************************************
//	***in this program backtracking method 
//	***************************************************************
    Misfit_pre=misfit;

    ini_step(dn_vp,np,un0_vp);
//    ini_step(dn_vs,np,un0_vs);


    Misfit_new=1.0e+20;
    printf("Misfit_pre == %15.12f\n",Misfit_pre);
    printf("\n");

    while(Misfit_new>Misfit_pre)
    {

      update_model(vp,vs,rho,
                   vp_n,vs_n,
                   dn_vp,un0_vp,dn_vs,un0_vs,
                   ntp, ntz, ntx, pml);
      
      get_lame_constants(lambda,mu,lambda_plus_two_mu,vp,vs,rho,ntp);

      Misfit_new=0.0f;

      for(is=0;is<ns;is++)
      {

        fdtd_cpml_2d_GPU(ntx,ntz,ntp,nx,nz,pml,dx,dz,
                       rick,itmax,dt,
                       is,s_ix[is],s_iz[is],
                       rho,lambda,mu,lambda_plus_two_mu,
                       k_x,k_x_half,k_z,k_z_half,
                       a_x,a_x_half,a_z,a_z_half,
                       b_x,b_x_half,b_z,b_z_half,c,
                       inv_flag,
                       seismogram_vx_syn,seismogram_vz_syn,
                       vx_borders_up,vx_borders_bottom,
                       vx_borders_left,vx_borders_right,
                       vz_borders_up,vz_borders_bottom,
                       vz_borders_left,vz_borders_right
                       );
        if(Misfit_new>1.0f)
        {

           printf("Attention!!!,Misfit is too large!!!\n");

           sprintf(filename,"./output/%dwrong_Gradient_vp.dat",iter+1);
           fp=fopen(filename,"wb");
           fwrite(&Gradient_vp[0],sizeof(float),np,fp);
           fclose(fp);


           sprintf(filename,"./output/%dwrong_vp.dat",iter+1);
           fp=fopen(filename,"wb");
           for(ix=pml;ix<=ntx-pml-1;ix++)
           {
              for(iz=pml;iz<=ntz-pml-1;iz++)
              {
                 fwrite(&vp[iz*ntx+ix],sizeof(float),1,fp);
              }
           }
        
           fclose(fp);
           return;

        }
 
        if(MULTI_SCALE==1)
        {
           // READ IN OBSERVED SEISMOGRAMS...
           sprintf(filename,"./output/%dsource_seismogram_vx_obs_filted.dat",is+1);
           fp=fopen(filename,"rb");
           fread(&seismogram_vx_obs[0],sizeof(float),nx*itmax,fp);
           fclose(fp);

           sprintf(filename,"./output/%dsource_seismogram_vz_obs_filted.dat",is+1);
           fp=fopen(filename,"rb");
           fread(&seismogram_vz_obs[0],sizeof(float),nx*itmax,fp);
           fclose(fp);
        }

        else
        {
           // READ IN OBSERVED SEISMOGRAMS...
           sprintf(filename,"./output/%dsource_seismogram_vx_obs.dat",is+1);
           fp=fopen(filename,"rb");
           fread(&seismogram_vx_obs[0],sizeof(float),nx*itmax,fp);
           fclose(fp);

           sprintf(filename,"./output/%dsource_seismogram_vz_obs.dat",is+1);
           fp=fopen(filename,"rb");
           fread(&seismogram_vz_obs[0],sizeof(float),nx*itmax,fp);
           fclose(fp);
        }

        for(ip=0;ip<nx*itmax;ip++)
        {
           seismogram_vx_rms[ip]=seismogram_vx_obs[ip]-seismogram_vx_syn[ip];
           seismogram_vz_rms[ip]=seismogram_vz_obs[ip]-seismogram_vz_syn[ip];
        }
/*
      sprintf(filename,"./output/%diter_seismogram_vx_rms_update.dat",iter+1);
      fp=fopen(filename,"wb");
      fwrite(&seismogram_vx_rms[0],sizeof(float),nx*itmax,fp);
      fclose(fp);

      sprintf(filename,"./output/%diter_seismogram_vz_rms_update.dat",iter+1);
      fp=fopen(filename,"wb");
      fwrite(&seismogram_vz_rms[0],sizeof(float),nx*itmax,fp);
      fclose(fp);
*/
      
        for(ip=0;ip<nx*itmax;ip++)
        {
           Misfit_new=Misfit_new+
           (seismogram_vx_rms[ip]*seismogram_vx_rms[ip])+
           (seismogram_vz_rms[ip]*seismogram_vz_rms[ip]);
        }
      }

      Misfit_new=Misfit_new*1.0e+10;
      printf("Misfit_new == %15.12f\n",Misfit_new);

      *un0_vp=*un0_vp/2.0f;
    }


    free(seismogram_vx_obs);
    free(seismogram_vz_obs);
    free(seismogram_vx_rms);
    free(seismogram_vz_rms);

    free(dn_vp);
    free(dn_vs);

    free(un0_vp);
    free(un0_vs);


    free(Gradient_vp);

    free(Gradient_vs_all);
    free(Gradient_vs);
//    free(Gradient_vp_all);


  return;
  }


//*************************************************************************//*******un0*cnmax=vmax*0.01
//************************************************************************

 void ini_step(float *dn, int np, float *un0)
 {
    float vpmax=5500.0f;
    float dnmax=-1.0e20f;
    int ip;

    for(ip=0;ip<np;ip++)
    {
        if(dnmax<abs(dn[ip]))
        {
           dnmax=abs(dn[ip]);
        }
    }
    
    *un0=vpmax*0.01f/dnmax;

  return;
  }


/*=========================================================================
  To calculate the updated model...
  ========================================================================*/

  void update_model(float *vp, float *vs, float *rho, 
                    float *vp_n, float *vs_n,
                    float *dn_vp, float *un_vp, float *dn_vs, float *un_vs,
                    int ntp, int ntz, int ntx, int pml)
  {
    int ip,ipp;
    int iz,ix;
    int nx=ntx-2*pml;

    for(iz=pml;iz<=ntz-pml-1;iz++)
    {
        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
           ip=iz*ntx+ix;
           ipp=(iz-pml)*nx+ix-pml;
           vp[ip]=vp_n[ip]+*un_vp*dn_vp[ipp];
//           vs[ip]=vs_n[ip]+*un_vs*dn_vs[ipp];
        }
    }

//  Model in PML..............

    for(iz=0;iz<=pml-1;iz++)
    {

        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+pml;

            vp[ip]=vp[ipp];
//            vs[ip]=vs[ipp];
        }

        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+ix;
           
//            vp[ip]=vp[ipp];
            vp[ip]=vp[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+ntx-pml-1;

            vp[ip]=vp[ipp];
//            vs[ip]=vs[ipp];
        }
    }

    for(iz=pml;iz<=ntz-pml-1;iz++)
    {
        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=iz*ntx+pml;
            
//            vp[ip]=vp[ipp];
            vp[ip]=vp[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=iz*ntx+ntx-pml-1;
       
            vp[ip]=vp[ipp];
//            vs[ip]=vs[ipp];
        }

     }

     for(iz=ntz-pml;iz<ntz;iz++)
     {
         
        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+pml;

            vp[ip]=vp[ipp];
//            vs[ip]=vs[ipp];
        }

        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+ix;
           
            vp[ip]=vp[ipp];
//            vs[ip]=vs[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+ntx-pml-1;

            vp[ip]=vp[ipp];
//            vs[ip]=vs[ipp];
        }
    }


/*
    for(iz=0;iz<=pml-1;iz++)
    {

        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+pml;

            vs[ip]=vs[ipp];
        }

        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+ix;
           
            vs[ip]=vs[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+ntx-pml-1;

            vs[ip]=vs[ipp];
        }
    }

    for(iz=pml;iz<=ntz-pml-1;iz++)
    {
        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=iz*ntx+pml;
            
            vs[ip]=vs[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=iz*ntx+ntx-pml-1;
       
            vs[ip]=vs[ipp];
        }

     }

     for(iz=ntz-pml;iz<ntz;iz++)
     {
         
        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+pml;

            vs[ip]=vs[ipp];
        }

        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+ix;
           
            vs[ip]=vs[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+ntx-pml-1;

            vs[ip]=vs[ipp];
        }
    }
*/

  return;
  }


/***********************************************************************
!                initial model
!***********************************************************************/
  void ini_model_mine(float *vp, float *vp_n, int ntp, int ntz, int ntx, int pml, int flag)
  {
/*  flag == 1 :: P velocity
    flag == 2 :: S velocity
    flag == 3 :: Density
*/
    int window=10;
    float *vp_old1;

    float sum;
    int number;

    int iz,ix;
    int izw,ixw;
    int ip,ipp;

    vp_old1=(float*)malloc(sizeof(float)*ntp);


    for(ip=0;ip<ntp;ip++)
    {
        vp_old1[ip]=vp[ip];
    }

//-----smooth in the x direction---------
    
    for(iz=pml;iz<=ntz-pml-1;iz++)
    {
        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            sum=0.0f;
            number=0;
            for(izw=iz-window;izw<iz+window;izw++)
            {
                for(ixw=ix-window;ixw<ix+window;ixw++)
                {

                    ip=izw*ntx+ixw;
                    sum=sum+vp_old1[ip];
                    number=number+1;
                }
            }
            ip=iz*ntx+ix;
            vp[ip]=sum/number;
         }
    }    

    for(iz=pml;iz<=pml+12;iz++)
    {
        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            ip=iz*ntx+ix;
            if(flag==1) 
            {
               vp[ip]=1500.0;
            }
            
            if(flag==2)
            {
               vp[ip]=0.0;
            }
      
            if(flag==3)
            {
               vp[ip]=1000.0;
            }
        }
    }

//  Model in PML..............

    for(iz=0;iz<=pml-1;iz++)
    {

        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+pml;

            vp[ip]=vp[ipp];
        }

        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+ix;
           
            vp[ip]=vp[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=pml*ntx+ntx-pml-1;

            vp[ip]=vp[ipp];
        }
    }

    for(iz=pml;iz<=ntz-pml-1;iz++)
    {
        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=iz*ntx+pml;
            
            vp[ip]=vp[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=iz*ntx+ntx-pml-1;
       
            vp[ip]=vp[ipp];
        }

     }

     for(iz=ntz-pml;iz<ntz;iz++)
     {
         
        for(ix=0;ix<=pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+pml;

            vp[ip]=vp[ipp];
        }

        for(ix=pml;ix<=ntx-pml-1;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+ix;
           
            vp[ip]=vp[ipp];
        }

        for(ix=ntx-pml;ix<ntx;ix++)
        {
            ip=iz*ntx+ix;
            ipp=(ntz-pml-1)*ntx+ntx-pml-1;

            vp[ip]=vp[ipp];
        }
    }

    for(ip=0;ip<ntp;ip++)
    {
        vp_n[ip]=vp[ip];
    }

    free(vp_old1);
  }


/*=======================================================================

  subroutine preprocess(nz,nx,dx,dz,P)

!=======================================================================*/
// in this program Precondition P is computed
  
  void Preprocess(int nz, int nx, float dx, float dz, float *P)
  {
     int iz,iz_depth_one,iz_depth_two;
     float z,delta1,a,temp,z1,z2;
  
     a=3.0f;
     iz_depth_one=10;
     iz_depth_two=15;
     delta1=(iz_depth_two-iz_depth_one)*dx;
     z1=(iz_depth_one-1)*dz;
     z2=(iz_depth_two-1)*dz;
  
     for(iz=0;iz<nz;iz++)
     { 
         z=iz*dz;
         if(z>=0.0f&&z<=z1)
         {
            P[iz]=0.0f;
         }
     
         if(z>z1&&z<=z2)
         {
            temp=z-z1-delta1;
            temp=a*temp*2/delta1;
            temp=temp*temp;
            P[iz]=exp(-0.5f*temp);
         }
    
         if(z>z2)
         {
            P[iz]=float(z)/float(z2)*1.5;
         }
      }
      for(iz=0;iz<nz;iz++)
      {
          z=iz*dz;
          if(z>=0.0&&z<=z1)
          {
             P[iz]=0.0f;
          }
          if(z>z1)
          {
             P[iz]=1.0f;
          }
      }
  }

void filter_of_ricker(
     float fpp, float fs, float *rick, 
     int itmax, float *rick_filted, float dt)

{
     
    int  it;

/*=========================================================
  Parameters of the filter...
  ========================================================*/
    int   K,N,NFFT;
    int   window_flag;

    float df,fc,f_total,wp,ws,wc,d_w;
          // window_flag==1 Hanning window;
          // window_flag==2 Blackman-Harris window

    float *win;
    float *hd;
    float *hhh;

    float *rick_fft_real;
    float *rick_fft_imag;

    float *rick_new;
    float *data_out;
    
/*=========================================================
  Calculate the number of fft points...
  ========================================================*/
    window_flag=1;
    K=ceil(log(1.0*itmax)/log(2.0));
    NFFT=int(pow(2.0,K));

//------------Calculate the bandwidth of the filter ---------------

    df=1.0/(NFFT*dt);
    fc=(fpp+fs)/3.0;

//------------Transfer angle frequency to frequency ---------------

    f_total=(NFFT-1)*df;           //the maxum of frequency
    wp=(2*PI*fpp)/f_total;
    ws=(2*PI*fs)/f_total;	
    wc=(2*PI*fc)/f_total;

//------------Calculate the width of the window -------------------

    d_w=ws-wp;
    N=ceil(12.0*PI/d_w)+1;

    win=(float*)malloc(sizeof(float)*N);
    hd =(float*)malloc(sizeof(float)*N);
    hhh=(float*)malloc(sizeof(float)*N);

    rick_fft_real=(float*)malloc(sizeof(float)*NFFT);
    rick_fft_imag=(float*)malloc(sizeof(float)*NFFT);

    rick_new=(float*)malloc(sizeof(float)*NFFT);
    data_out=(float*)malloc(sizeof(float)*NFFT);

/*##################################################################
  -------------------   The Part Of Filter   -----------------------
  ################################################################*/

//--------------------window function-------------------------------

    if(window_flag==1)
    {
       Hamming(N,win);
    }

    if(window_flag==2)
    {
       Blackman_harris(N,win);
    }

//------------------pulse response-------------------------------

    hdn(hd,N,wc);

//-----------------lower filter----------------------------------
	
    hh(win,hd,hhh,N);

    if(itmax<NFFT)
    {
       for(it=0;it<itmax;it++)
       {
           rick_new[it]=rick[it];
       }
    }

    con(rick_new,NFFT,hhh,N,data_out);

    if(itmax<NFFT)
    {
       for(it=0;it<itmax;it++)
       {
           rick_filted[it]=data_out[it];
       }
    }

    free(win);
    free(hd);
    free(hhh);
    free(rick_fft_real);
    free(rick_fft_imag);
    free(rick_new);
    free(data_out);

    return;
   }


void filter_of_ricker_frequency(
     float fpp, float fs, float *rick, 
     int itmax, float *rick_filted, float dt)

{
    int  it;
/*=========================================================
  Parameters of the filter...
  ========================================================*/
    int   K,NFFT;

    float df;
          // window_flag==1 Hanning window;
          // window_flag==2 Blackman-Harris window

    float *rick_fft_real;
    float *rick_fft_imag;
    float *H,*P;
    
/*=========================================================
  Calculate the number of fft points...
  ========================================================*/
    K=ceil(log(1.0*itmax)/log(2.0));
    NFFT=int(pow(2.0,K));

//------------Calculate the bandwidth of the filter ---------------

    df=1.0/(NFFT*dt);

    rick_fft_real=(float*)malloc(sizeof(float)*NFFT);
    rick_fft_imag=(float*)malloc(sizeof(float)*NFFT);
    H=(float*)malloc(sizeof(float)*NFFT);
    P=(float*)malloc(sizeof(float)*NFFT);
    
/*##################################################################
  -------------------   The Part Of Filter   -----------------------
  ################################################################*/

    for(it=0;it<NFFT;it++)
    {
        rick_fft_real[it]=0.0;
        rick_fft_imag[it]=0.0;
    }

    for(it=0;it<itmax;it++)
    {
        rick_fft_real[it]=rick[it];
    }

    fft(rick_fft_real, rick_fft_imag, NFFT, 1);

    fre_filter(H,NFFT,fpp,fs,df);    fre_filter_P(P,NFFT,fpp,fs,df);

    for(it=0;it<NFFT;it++)
    {
        H[it]=H[it]*P[it];
    }
// ----------use the frequency filter ----------------------------
    for(it=0;it<NFFT;it++)
    {
        rick_fft_real[it]=rick_fft_real[it]*H[it];
        rick_fft_imag[it]=rick_fft_imag[it]*H[it];
    }

    fft(rick_fft_real, rick_fft_imag, NFFT, -1);

    for(it=0;it<itmax;it++)
    {
        rick_filted[it]=rick_fft_real[it];
    }

    free(rick_fft_real);
    free(rick_fft_imag);
    free(P);
    free(H);

    return;
   }


/*==============================================================================

  Filter of the observed seismograms

  ============================================================================*/
//******************************************************************************

void filter_of_seismogram(
     float fpp, float fs, float **seismogram, 
     int itmax, int ntr, float **seismogram_filted, float dt)

{
    int  it,itr;

/*=========================================================
  Parameters of the filter...
  ========================================================*/

    int   K,N,NFFT;
    int   window_flag;

    float df,fc,f_total,wp,ws,wc,d_w;

    float *win;
    float *hd;
    float *hhh;

    float *rick_fft_real;
    float *rick_fft_imag;

    float *rick_new;
    float *data_out;
    
/*=========================================================
  Calculate the number of fft points...
  ========================================================*/
    window_flag=1;
    K=ceil(log(1.0*itmax)/log(2.0));
    NFFT=int(pow(2.0,K));

    printf("NFFT=%d\n",NFFT);

//------------Calculate the bandwidth of the filter ---------------

    df=1.0/(NFFT*dt);
    fc=(fpp+fs)/3.0;

//------------Transfer angle frequency to frequency ---------------

    f_total=(NFFT-1)*df;           //the maxum of frequency
    wp=(2*PI*fpp)/f_total;
    ws=(2*PI*fs)/f_total;	
    wc=(2*PI*fc)/f_total;

//------------Calculate the width of the window -------------------

    d_w=ws-wp;
    N=ceil(12.0*PI/d_w)+1;

    win=(float*)malloc(sizeof(float)*N);
    hd =(float*)malloc(sizeof(float)*N);
    hhh=(float*)malloc(sizeof(float)*N);

    rick_fft_real=(float*)malloc(sizeof(float)*NFFT);
    rick_fft_imag=(float*)malloc(sizeof(float)*NFFT);

    rick_new=(float*)malloc(sizeof(float)*NFFT);
    data_out=(float*)malloc(sizeof(float)*NFFT);

/*##################################################################
  -------------------   The Part Of Filter   -----------------------
  ################################################################*/

//--------------------window function-------------------------------

    if(window_flag==1)
    {
       Hamming(N,win);
    }

    if(window_flag==2)
    {
       Blackman_harris(N,win);
    }

//------------------pulse response-------------------------------

    hdn(hd,N,wc);

//-----------------lower filter----------------------------------
	
    hh(win,hd,hhh,N);

    for(itr=0;itr<ntr;itr++)
    {
        if(itmax<NFFT)
        {
           for(it=0;it<itmax;it++)
           {
               rick_new[it]=seismogram[it][itr];
           }
        }

        con(rick_new,NFFT,hhh,N,data_out);

        for(it=0;it<itmax;it++)
        {
            seismogram_filted[it][itr]=data_out[it];
        }
    }

    free(win);
    free(hd);
    free(hhh);
    free(rick_fft_real);
    free(rick_fft_imag);
    free(rick_new);
    free(data_out);

    return;
   }


//******************************************************************************

void filter_of_seismogram_frequency(
     float fpp, float fs, float **seismogram, 
     int itmax, int ntr, float **seismogram_filted, float dt)
{
    int  it;
/*=========================================================
  Parameters of the filter...
  ========================================================*/
    int   K,NFFT;

    float df;

    float *rick_fft_real;
    float *rick_fft_imag;

    float *P,*H;

    int itr;

/*=========================================================
  Calculate the number of fft points...
  ========================================================*/
    K=ceil(log(1.0*itmax)/log(2.0));
    NFFT=int(pow(2.0,K));

//------------Calculate the bandwidth of the filter ---------------

    df=1.0/(NFFT*dt);

    rick_fft_real=(float*)malloc(sizeof(float)*NFFT);
    rick_fft_imag=(float*)malloc(sizeof(float)*NFFT);

    H=(float*)malloc(sizeof(float)*NFFT);
    P=(float*)malloc(sizeof(float)*NFFT);
    
/*##################################################################
  -------------------   The Part Of Filter   -----------------------
  ################################################################*/

    for(itr=0;itr<ntr;itr++)
    {
        for(it=0;it<NFFT;it++)
        {
            rick_fft_real[it]=0.0;
            rick_fft_imag[it]=0.0;
        }
        for(it=0;it<itmax;it++)
        {
            rick_fft_real[it]=seismogram[it][itr];
        }

        fft(rick_fft_real, rick_fft_imag, NFFT, 1);

        fre_filter(H,NFFT,fpp,fs,df);        fre_filter_P(P,NFFT,fpp,fs,df);

        for(it=0;it<NFFT;it++)
        {
            H[it]=H[it]*P[it];
        }
// ----------use the frequency filter ----------------------------
        for(it=0;it<NFFT;it++)
        {
            rick_fft_real[it]=rick_fft_real[it]*H[it];
            rick_fft_imag[it]=rick_fft_imag[it]*H[it];
        }

        fft(rick_fft_real, rick_fft_imag, NFFT, -1);

        for(it=0;it<itmax;it++)
        {
            seismogram_filted[it][itr]=rick_fft_real[it];
        }
    }

    free(rick_fft_real);
    free(rick_fft_imag);
    free(P);
    free(H);

    return;
   }
/*=================================================================
	
! ----------------------------------------------------------------
!  This subroutine is used for yeilding a hamming window function. 
!  N    : length of the window
!  win  : the window window
!  2012.3.23/Jie Wang
! ----------------------------------------------------------------

*=================================================================*/
   void Hamming(int N,float *win)
   {

	int   NN,i;
        float a1,a2;

	a1=0.54;
	a2=0.46;
	NN=N-1;

        for(i=0;i<N;i++)
        {
            win[i]=a1-a2*cos(2.0*PI*i/NN);
        }
	
	return;
   }

/*==================================================================

! --------------------------------------------------------------
!  This subroutine is used for yeilding a B-H window function. 
!  N    : length of the window
!  win  : the window window
!  2012.3.23/Jie Wang
! --------------------------------------------------------------

*==================================================================*/
   void Blackman_harris(int N,float *win)
   {		
        int NN,i;	
	float a1,a2;
        float a3,a4;

	a1=0.35875;
	a2=0.48829;
	a3=0.14128;
	a4=0.01168;

	NN=N-1;
        for(i=0;i<N;i++)
        {
            win[i]=a1-a2*cos(2.0*PI*i/NN)+a3*cos(4.0*PI*i/NN)-a4*cos(6.0*PI*i/NN);
        }

	return;
   }
  

/*=====================================================================

!---------------------------------------------------------------
!   	This subroutine is used for yeilding a respose of a pulse.
!   	N    : length of the window
!   	wc   : œØÖ¹ÆµÂÊ
!   	hd   : response of a pulse
!   	2012.3.23/Jie Wang
! --------------------------------------------------------------

*=====================================================================*/
   void hdn(float *hd,int N,float wc)
   {
        int alpha;
	int i;
        float m;

	alpha=(N-1)/2;

        for(i=0;i<N;i++)
        {
            m=(i-alpha+1.0e-10);
            hd[i]=sin(wc*m)/(PI*m);
        }

        return;
    }
	
/*====================================================================

! --------------------------------------------------------------
!   	This subroutine is used for yeilding a lower pass filter 
!   	N    : length of the window
!	win  : window function
!   	hd   : response of a pulse
!   	h    : lower pass filtenum2str(num,asc)r
!   	2012.3.23/Jie Wang
! --------------------------------------------------------------

*====================================================================*/

   void hh(float *win, float *hd, float *h, int N)
   {
        int i;

        for(i=0;i<N;i++)
        {
            h[i]=hd[i]*win[i];
        }

	return;
   }
/*====================================================================*/

/*=====================================================================

!	--------------------------------------------------------------
!   	This subroutine is used for calculate the con in time domain. 
!   	X(m) : input data
!   	H(n) : filter
!   	Y(l) : output
!   	2012.3.28/Jie Wang
!   	--------------------------------------------------------------

*=====================================================================*/

   void con(float *x, int m, float *h, int n, float *out)
   {
	float *y;
	int l,n_half;
        int i,k;
	l=m+n-1;
        y=(float*)malloc(l*sizeof(float));
	for(k=0;k<l;k++)
        {
		y[k]=0.0;
		for(i=0;i<m;i++)
                {
			if((k-i)>=0&&(k-i)<=(n-1))
                        {
				y[k]=y[k]+x[i]*h[k-i];
			}
		}
	}

	n_half=n/2;

        for(i=0;i<m;i++)
        {
            out[i]=y[i+n_half];
        }
        free(y);
	return;
   }

/*======================================================================

!	--------------------------------------------------------------
!   	This subroutine is used for yeilding a filter in frequency. 
!   	H    : the real part of the filter,while the image part is 0.
!   	NFFT : number of the fft points
!   	w1   : The start frequency
!   	w2   : The end frequency
!   	df   : the sampling interval in frequency
!   	2012.3.22/Jie Wang
!   	--------------------------------------------------------------

*======================================================================*/

   void fre_filter(float *H, int NFFT, float w1, float w2, float df)   {
        int i;
        float f;

	for(i=0;i<NFFT/2+1;i++)
        {
		f=i*df;
		if(f<=w2&&f>=w1)
                {
			H[i]=1.0;
                }
		else
                {
			H[i]=0.0;
                }
        }

	for(i=NFFT/2+1;i<NFFT;i++)
        {
		f=i*df;
		H[i]=H[NFFT+1-i];  // maybe has some problem...
        }
	
	return;
    }

/* ===================================================================
!	--------------------------------------------------------------
!   	This subroutine is used for yeilding a filter in frequency. 
!   	P    : the Gaussian taper 
!   	NFFT : number of the fft points
!   	f1   : The start descrease frequency
!   	f2   : The ending frequency
!   	df   : the sampling interval in frequency
!   	2012.7.30/Jie Wang
!   	--------------------------------------------------------------
!* =================================================================*/

    void fre_filter_P(float *P, int NFFT, float f1, float f2, float df)
    {
	float f;
        float delta_f,a,temp1,temp2,temp3;
        int it;

	delta_f=f2-f1;
	a=1.6;
//        a=2.0;
	
	for(it=0;it<NFFT;it++)
        {
	   f=it*df;
	   if(f<=f1)
           {
	      P[it]=1.0;
           }
           if(f>f1&&f<=f2)
           {
	      temp1=f-f1;
	      temp2=delta_f/2.0;
	      temp3=a*temp1/temp2;
	      P[it]=exp(-0.5*temp3*temp3);
           }
           if(f>f2)
           {
	      P[it]=0.0;
           }
	}
	     
	return;
    }

/*==============================================================

  This subroutine is used for FFT/IFFT
   
===========================================================*/
void fft(float *xreal,float *ximag,int n,int sign)
{
	int i,j,k,m,temp;
	int h,q,p;
	float t;
	float *a,*b;
	float *at,*bt;
	int *r;
	
	//xreal=(float*)malloc(n*sizeof(float));
	//ximag=(float*)malloc(n*sizeof(float));
	a=(float*)malloc(n*sizeof(float));
	b=(float*)malloc(n*sizeof(float));
	r=(int*)malloc(n*sizeof(int));
	at=(float*)malloc(n*sizeof(float));
	bt=(float*)malloc(n*sizeof(float));

	m=(int)(log(n-0.5)/log(2.0))+1; //2的幂，2的m次方等于n；
	for(i=0;i<n;i++)
	{
		a[i]=xreal[i];
		b[i]=ximag[i];
		r[i]=i;
	}
	for(i=0,j=0;i<n-1;i++)  //0到n的反序；
	{
		if(i<j)
		{
			temp=r[i];
			r[i]=j;
			r[j]=temp;
		}
		k=n/2;
		while(k<(j+1))
		{
			j=j-k;
			k=k/2;
		}
		j=j+k;
	}

	t=2*PI/n;
	for(h=m-1;h>=0;h--)
	{
		p=(int)pow(2.0,h);
		q=n/p;
		for(k=0;k<n;k++)
		{
			at[k]=a[k];
			bt[k]=b[k];
		}
		
		for(k=0;k<n;k++)
		{
			if(k%p==k%(2*p))
			{

				a[k]=at[k]+at[k+p];
				b[k]=bt[k]+bt[k+p];
				a[k+p]=(at[k]-at[k+p])*cos(t*(q/2)*(k%p))-(bt[k]-bt[k+p])*sign*sin(t*(q/2)*(k%p));
				b[k+p]=(bt[k]-bt[k+p])*cos(t*(q/2)*(k%p))+(at[k]-at[k+p])*sign*sin(t*(q/2)*(k%p));
			}
		}

	}

	for(i=0;i<n;i++)
	{
		if(sign==1)
		{
			xreal[r[i]]=a[i];
			ximag[r[i]]=b[i];
		}
		else if(sign==-1)
		{
			xreal[r[i]]=a[i]/n;
			ximag[r[i]]=b[i]/n;
		}
	}

	free(a);
	free(b);
	free(r);
	free(at);
	free(bt);
}


/*=====================================================================*/


void get_wiener_filter(float *rick_fft_real, float *rick_fft_imag, 
                       float *rick_target_fft_real, float *rick_target_fft_imag,
                       float *filter_wiener_fft_real,
                       float *filter_wiener_fft_imag, int NFFT)
{

    float rick_amp;
    int it;

      //---Calculate the Wiener filter.....
      for(it=0;it<NFFT;it++)
      {
          rick_amp=pow(rick_fft_real[it],2)+pow(rick_fft_imag[it],2)+EPS;

          filter_wiener_fft_real[it]=(rick_fft_real[it]*rick_target_fft_real[it]+
                                      rick_fft_imag[it]*rick_target_fft_imag[it])/rick_amp;
          filter_wiener_fft_imag[it]=(rick_fft_real[it]*rick_target_fft_imag[it]-
                                      rick_fft_imag[it]*rick_target_fft_real[it])/rick_amp;
      }

      return;
}

void filter_of_ricker_wiener(float *rick_fft_real, float *rick_fft_imag,
                             float *filter_wiener_fft_real,
                             float *filter_wiener_fft_imag,
                             int NFFT, float *rick, int itmax)
{
     int it;
     float *rick_filted_fft_real,*rick_filted_fft_imag;

     rick_filted_fft_real=(float*)malloc(sizeof(float)*NFFT);
     rick_filted_fft_imag=(float*)malloc(sizeof(float)*NFFT);

   
     //----Filte the original ricker wave...

     for(it=0;it<NFFT;it++)
     {
         rick_filted_fft_real[it]=rick_fft_real[it]*filter_wiener_fft_real[it]+
                                  rick_fft_imag[it]*filter_wiener_fft_imag[it];
         rick_filted_fft_imag[it]=rick_fft_real[it]*filter_wiener_fft_imag[it]+
                                  rick_fft_imag[it]*filter_wiener_fft_real[it];
     }

     for(it=0;it<itmax;it++)
     {
         rick[it]=0.0;
     }

     fft(rick_filted_fft_real, rick_filted_fft_imag, NFFT, -1);

     for(it=0;it<itmax;it++)
     {
       rick[it]=rick_filted_fft_real[it];
     }

     free(rick_filted_fft_real);
     free(rick_filted_fft_imag);
     return;

}
     

void filter_of_seismogram_wiener(
     float **seismogram, int itmax, int ntr, float **seismogram_filted,
     float *filter_wiener_fft_real, float *filter_wiener_fft_imag, int NFFT)
{
     int it,itr;

     float *seismogram_fft_real;
     float *seismogram_fft_imag;

     float *seismogram_filted_fft_real;
     float *seismogram_filted_fft_imag;

     seismogram_fft_real=(float*)malloc(sizeof(float)*NFFT);
     seismogram_fft_imag=(float*)malloc(sizeof(float)*NFFT);
     seismogram_filted_fft_real=(float*)malloc(sizeof(float)*NFFT);
     seismogram_filted_fft_imag=(float*)malloc(sizeof(float)*NFFT);

     for(itr=0;itr<ntr;itr++)
     {
        
        for(it=0;it<NFFT;it++)
        {
            seismogram_fft_real[it]=0.0;
            seismogram_fft_imag[it]=0.0;
        }
        for(it=0;it<itmax;it++)
        {
            seismogram_fft_real[it]=seismogram[it][itr];
        }

        fft(seismogram_fft_real, seismogram_fft_imag, NFFT, 1);
     
        //----Filte the observed seismograms...

        for(it=0;it<NFFT;it++)
        {
            seismogram_filted_fft_real[it]=seismogram_fft_real[it]*filter_wiener_fft_real[it]+
                                           seismogram_fft_imag[it]*filter_wiener_fft_imag[it];
            seismogram_filted_fft_imag[it]=seismogram_fft_real[it]*filter_wiener_fft_imag[it]+
                                           seismogram_fft_imag[it]*filter_wiener_fft_real[it];
        }

        fft(seismogram_filted_fft_real, seismogram_filted_fft_imag, NFFT, -1);

        for(it=0;it<itmax;it++)
        {
            seismogram_filted[it][itr]=seismogram_filted_fft_real[it];
        }
      }

      free(seismogram_fft_real);
      free(seismogram_fft_imag);
      free(seismogram_filted_fft_real);
      free(seismogram_filted_fft_imag);
      return;
}   
          
