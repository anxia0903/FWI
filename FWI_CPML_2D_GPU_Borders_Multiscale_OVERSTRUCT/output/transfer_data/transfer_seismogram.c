#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

int main()
{

/*=========================================================
  Parameters of Cartesian coordinate...
  ========================================================*/

  FILE *fp; 
  char filename[230];
  
  int nx=384;

  int   it,ix,ip;
  int   itmax=2500;

/*=========================================================
  Parameters of Sources and Receivers...
  ========================================================*/
  int is;
  int ns=10;


  float *seismogram_vx_obs, *seismogram_vz_obs;
  float **seis_vx_obs,**seis_vz_obs;



  seismogram_vx_obs=(float*)malloc(sizeof(float)*itmax*nx);
  seismogram_vz_obs=(float*)malloc(sizeof(float)*itmax*nx);

  seis_vx_obs=(float**)malloc(sizeof(float*)*itmax);
  seis_vz_obs=(float**)malloc(sizeof(float*)*itmax);

  for(it=0;it<itmax;it++)
  {
      seis_vx_obs[it]=(float*)malloc(sizeof(float)*nx);
      seis_vz_obs[it]=(float*)malloc(sizeof(float)*nx);
  }


  for(is=0;is<ns;is++)
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

      

      for(it=0;it<itmax;it++)
      {
          for(ix=0;ix<nx;ix++)
          {
              ip=it*nx+ix;
              seis_vx_obs[it][ix]=seismogram_vx_obs[ip];
              seis_vz_obs[it][ix]=seismogram_vz_obs[ip];
          }
      }
/*
      printf("2222222222222\n"); 
      for(it=0;it<itmax;it++)
      {
          for(ix=0;ix<nx;ix++)
          {
              seismogram_2d_T[ix][it]=seismogram_2d[it][ix];
          }
      }
      printf("3333333333333\n"); 
   
      for(ix=0;ix<nx;ix++)
      {
          for(it=0;it<itmax;it++)
          {
              ip=ix*itmax+it;
              seismogram_T[ip]=seismogram_2d_T[ix][it];
          }
      }

      // output OBSERVED SEISMOGRAMS...
      sprintf(filename,"../%dsource_seismogram_vx_obs_transfer.dat",is+1);
      fp=fopen(filename,"wb");
      fwrite(&seismogram_T[0],sizeof(float),nx*itmax,fp);
      fclose(fp);
*/

  }

}

  



  

