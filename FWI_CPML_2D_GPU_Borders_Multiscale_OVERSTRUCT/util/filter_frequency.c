
/*******************************************************************************

         --------------------------------------------------------------
        ! This program is used for testing the time domain and frequency 
        ! domain filter. 
        !
        ! In frequency domain the method can be found in the book of Li 
        ! zhen chun.
        ! In time domain two windows are added in the process.1-Hamming w
        ! window.2-Blackman-Harris window.
        !
        ! Parameter descriPItion
        !
        ! itmax     : maxum number of time of ricker wavelet 
        ! t0        : delay time of ricker wavelet
        ! f0	      : domainant frequency
        ! dt        : sampling interval of time
        !
        ! NFFT      : number of FFT
        ! N         : length of the window
        ! df        : sampling interval of frequency
        ! H(:)      : filter in frequency
        ! win(:)    : window function 
        ! win_flag  : ==1,Hamming window; ==2, Blackman-Harris window
        ! hd(:)     : pulse response
        ! hhh(:)    : filter in time domain
        ! data_out  : data that is filted by the fielter
        !	
        ! fp        :
        ! fs        :
        ! fc        :
        ! wp        :
        ! ws        :
        ! w_c       :
        ! d_w       : the parameters can be found in the book of 
        !             <Diginal signal of processing>,2006.page 
        !             276.Cheng peiqing.
        !                    	
        ! 2012.3.28/Jie Wang
        !
        !--------------------------------------------------------------

*******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PI 3.1415927
#define SIG_LEN 256

typedef struct {
    float real;
    float image;
} Complex;

Complex ComMul(Complex p1, Complex p2);
Complex ComAdd(Complex p1, Complex p2);
Complex ComSub(Complex p1, Complex p2);

int Reform(float** in, int len);
int BitRevers(int src, int size);
void Displace(float* in, int size, int M);
void CalcW(Complex* w, int len);
Complex* fft_1d(float* in, int len);

void ricker_wave(float* rick, int itmax, float f0, float t0, float dt);
void Hamming(int N, float* win);
void Blackman_harris(int N, float* win);
void hdn(float* hd, int N, float wc);
void hh(float* win, float* hd, float* h, int N);
void con(float* x, int m, float* h, int n, float* out);

void fre_filter(float* H, int NFFT, float w1, float w2, float df);
void fre_filter_P(float* P, int NFFT, float f1, float f2, float df);
void fft(float* xreal, float* ximag, int n, int sign);

int main()
{
    int slen, it;

    FILE* fp;
    float* data;
    Complex* out;
    /*=========================================================
  Parameters of ricker wave...
  ========================================================*/

    int itmax = 1500;
    float* rick;
    float f0 = 20.0f;
    float t0 = 0.1f;
    float dt = 5.0e-4f;

    /*=========================================================
  Parameters of the filter...
  ========================================================*/
    float f_filted;
    float* rick_filted;
    int K, N, NFFT;
    int window_flag;
    int dimension_flag = 1;

    float df, fpp, fs, fc, f_total, wp, ws, wc, d_w;
    float real_num = 2;
    // window_flag==1 Hanning window;
    // window_flag==2 Blackman-Harris window

    float* win;
    float* hd;
    float* hhh;

    float* rick_fft_real;
    float* rick_fft_imag;

    float* rick_new;
    float* data_out;

    float *H, *P;

    /*=========================================================
  Calculate the number of fft points...
  ========================================================*/
    window_flag = 1;
    K = ceil(log(1.0 * itmax) / log(2.0));
    NFFT = pow(2, K);

    //------------Calculate the bandwidth of the filter ---------------

    df = 1.0 / (NFFT * dt);
    fpp = 0;
    fs = 10;
    fc = (fpp + fs) / 3.0;

    //------------Transfer angle frequency to frequency ---------------

    f_total = (NFFT - 1) * df; //the maxum of frequency
    wp = (2 * PI * fpp) / f_total;
    ws = (2 * PI * fs) / f_total;
    wc = (2 * PI * fc) / f_total;

    //------------Calculate the width of the window -------------------

    d_w = ws - wp;
    N = ceil(12.0 * PI / d_w) + 1;

    //------------Allocate memory to vectors---------------------------

    win = (float*)malloc(sizeof(float) * N);
    hd = (float*)malloc(sizeof(float) * N);
    hhh = (float*)malloc(sizeof(float) * N);

    rick_fft_real = (float*)malloc(sizeof(float) * NFFT);
    rick_fft_imag = (float*)malloc(sizeof(float) * NFFT);

    rick_new = (float*)malloc(sizeof(float) * NFFT);
    data_out = (float*)malloc(sizeof(float) * NFFT);

    rick = (float*)malloc(sizeof(float) * itmax);
    rick_filted = (float*)malloc(sizeof(float) * itmax);

    H = (float*)malloc(sizeof(float) * NFFT);
    P = (float*)malloc(sizeof(float) * NFFT);
    /*##################################################################
  -------------------   The Part Of Filter   -----------------------
  ################################################################*/
    /*
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

    for(it=0;it<NFFT;it++)
    {
        rick_fft_real[it]=0.0;
        rick_fft_imag[it]=0.0;
    }
	
    hh(win,hd,hhh,N);

    for(it=0;it<N;it++)
    {
        rick_fft_real[it]=hhh[it];
    }

    fft(rick_fft_real, rick_fft_imag, NFFT, 1);

      fp=fopen("rick_wave_amp.dat","w");    
      for(it=0;it<100;it++)
      {
        fprintf(fp,"%18f%18f\n",it*df,sqrt(pow(rick_fft_imag[it],2)+pow(rick_fft_real[it],2)));
      }   
      fclose(fp);
  
//*#####################################################################
//-----Calculate the ricker wavelet/input the data that needed to --
//-----be filtered -------------------------------------------------
//####################################################################
   if(dimension_flag==1)
   {

      for(it=0;it<NFFT;it++)
      {
          rick_fft_real[it]=0.0;
          rick_fft_imag[it]=0.0;
          rick_new[it]=0.0;
      }

      ricker_wave(rick,itmax,f0,t0,dt);

      fp=fopen("rick_wave.dat","w");    
      for(it=0;it<itmax;it++)
      {
        fprintf(fp,"%18f%18f\n",it*dt,rick[it]);
      }   
      fclose(fp);
    
      if(itmax<NFFT)
      {
         for(it=0;it<itmax;it++)
         {
             rick_new[it]=rick[it];
         }
      }

      for(it=0;it<itmax;it++)
      {
          rick_fft_real[it]=rick[it];
      }

      printf("N===%d\n",N);
// ----------------------filter applied--------------------------
      printf("===============here'''''''''''''''''''\n");


      con(rick_new,NFFT,hhh,N,data_out);


      for(it=0;it<201;it++)
      {
          printf("%d   %15.12f\n",it+1,data_out[it]);
      }


      printf("===============here2==================\n");

      if(itmax<NFFT)
      {
         for(it=0;it<itmax;it++)
         {
             rick_filted[it]=data_out[it];
         }
      }

      fp=fopen("rick_wave_filted.dat","w");    
      for(it=0;it<itmax;it++)
      {
        fprintf(fp,"%18f%18f\n",it*dt,rick_filted[it]);
      }   
      fclose(fp);



    for(it=0;it<NFFT;it++)
    {
        rick_fft_real[it]=0.0;
        rick_fft_imag[it]=0.0;
    }
	

    for(it=0;it<itmax;it++)
    {
        rick_fft_real[it]=rick_filted[it];
    }

    fft(rick_fft_real, rick_fft_imag, NFFT, 1);

      fp=fopen("rick_wave_filted_amp.dat","w");    
      for(it=0;it<100;it++)
      {
        fprintf(fp,"%18f%18f\n",it*df,sqrt(pow(rick_fft_imag[it],2)+pow(rick_fft_real[it],2)));
      }   
      fclose(fp);

    }
      

/*
=========================================================
  Allocate the memory of parameters of ricker wave...
  ========================================================
  

    out = fft_1d(rick,itmax);

    fp=fopen("rick_amp.dat","w");    
    for(it=0;it<100;it++)
    {
      fprintf(fp,"%18f%18f\n",it*dt,sqrt(pow(out[it].real,2)+pow(out[it].image,2)));
    }   
    fclose(fp);

*/

    ricker_wave(rick, itmax, f0, t0, dt);

    fp = fopen("rick_wave.dat", "w");
    for (it = 0; it < itmax; it++) {
        fprintf(fp, "%18f%18f\n", it * dt, rick[it]);
    }
    fclose(fp);

    if (itmax < NFFT) {
        for (it = 0; it < itmax; it++) {
            rick_new[it] = rick[it];
        }
    }

    for (it = 0; it < itmax; it++) {
        rick_fft_real[it] = rick[it];
    }

    fft(rick_fft_real, rick_fft_imag, NFFT, 1);

    // --------------filter in frequency domain ----------------------

    f_filted = 7.0;

    fre_filter(H, NFFT, 0.0, f_filted, df);
    fre_filter_P(P, NFFT, 0.0, f_filted, df);

    for (it = 0; it < NFFT; it++) {
        H[it] = H[it] * P[it];
    }

    // ----------use the frequency filter ----------------------------
    for (it = 0; it < NFFT; it++) {
        rick_fft_real[it] = rick_fft_real[it] * H[it];
        rick_fft_imag[it] = rick_fft_imag[it] * H[it];
    }

    fft(rick_fft_real, rick_fft_imag, NFFT, -1);

    fp = fopen("rick_wave_filted.dat", "w");
    for (it = 0; it < itmax; it++) {
        fprintf(fp, "%18f%18f\n", it * dt, rick_fft_real[it]);
    }
    fclose(fp);

    fft(rick_fft_real, rick_fft_imag, NFFT, 1);

    fp = fopen("rick_wave_filted_amp.dat", "w");
    for (it = 0; it < 100; it++) {
        fprintf(fp, "%18f%18f\n", it * df, sqrt(pow(rick_fft_imag[it], 2) + pow(rick_fft_real[it], 2)));
    }
    fclose(fp);

    free(rick);
    free(rick_filted);
    free(win);
    free(hd);
    free(hhh);
    free(rick_fft_real);
    free(rick_fft_imag);
    free(rick_new);
    free(data_out);
    free(H);
    free(P);
}

Complex ComMul(Complex p1, Complex p2)
{
    Complex res;
    res.real = p1.real * p2.real - p1.image * p2.image;
    res.image = p1.real * p2.image + p1.image * p2.real;

    return res;
}

Complex ComAdd(Complex p1, Complex p2)
{
    Complex res;
    res.real = p1.real + p2.real;
    res.image = p1.image + p2.image;

    return res;
}

Complex ComSub(Complex p1, Complex p2)
{
    Complex res;
    res.real = p1.real - p2.real;
    res.image = p1.image - p2.image;

    return res;
}

int BitRevers(int src, int size)
{
    int temp = src;
    int dst = 0;
    int i = 0;
    for (i = size - 1; i >= 0; i--) {
        dst = ((temp & 0x1) << i) | dst;
        temp = temp >> 1;
    }
    return dst;
}

void Displace(float* in, int size, int M)
{
    int i;
    int new_i;
    float t;
    for (i = 1; i < size; i++) {
        new_i = BitRevers(i, M);
        if (new_i > i) {
            t = in[i];
            in[i] = in[new_i];
            in[new_i] = t;
        }
    }
}
void CalcW(Complex* w, int N)
{
    int i;

    for (i = 0; i < N; i++) {
        w[i].real = cos(PI * i / N);
        w[i].image = -1 * sin(PI * i / N);
    }
}

int Reform(float** in, int len)
{
    int i = 0;
    int w = 1;

    while (w * 2 <= len) {
        w = w * 2;
        i++;
    }

    if (w < len) {

        *in = realloc(*in, w * 2 * sizeof(float));
        for (i = len; i < w * 2; i++)
            in[i] = 0;
        return i + 1;
    }
    return i;
}
Complex* fft_1d(float* data, int len)
{
    int M, i = 0, j, k;
    int GroupNum;
    int CellNum;
    int reallen;
    int pos1, pos2;
    float* in;
    Complex *res, *t;

    Complex *w, mul;

    in = data;
    M = Reform(&in, len); /*let the data length is pow of 2*/
    reallen = pow(2, M);
    res = malloc(reallen * sizeof(Complex));

    w = malloc(reallen * sizeof(Complex) / 2);
    CalcW(w, reallen / 2);

    Displace(in, reallen, M);
    while (i < reallen) {
        res[i].real = in[i];
        res[i].image = 0;
        i++;
    }

    GroupNum = reallen / 2;
    CellNum = 1;

    for (i = 0; i < M; i++) {
        for (j = 0; j < GroupNum; j++) {
            for (k = 0; k < CellNum; k++) {
                pos1 = j * CellNum * 2 + k;
                pos2 = pos1 + CellNum;

                mul = ComMul(res[pos2], w[k * GroupNum]);

                res[pos2] = ComSub(res[pos1], mul);
                res[pos1] = ComAdd(res[pos1], mul);
            }
        }
        GroupNum = GroupNum / 2;
        CellNum = CellNum * 2;
    }
    return res;
}

/*==========================================================

  This subroutine is used for calculating the ricker wave
   
===========================================================*/

void ricker_wave(float* rick, int itmax, float f0, float t0, float dt)
{
    int it;
    float temp;

    FILE* fp;

    for (it = 0; it < itmax; it++) {
        temp = PI * f0 * (it * dt - t0);
        temp = temp * temp;
        rick[it] = (1.0f - 2.0f * temp) * exp(-temp);
    }
}

/*=================================================================
	
! ----------------------------------------------------------------
!  This subroutine is used for yeilding a hamming window function. 
!  N    : length of the window
!  win  : the window window
!  2012.3.23/Jie Wang
! ----------------------------------------------------------------

*=================================================================*/
void Hamming(int N, float* win)
{

    int NN, i;
    float a1, a2;

    a1 = 0.54;
    a2 = 0.46;
    NN = N - 1;

    for (i = 0; i < N; i++) {
        win[i] = a1 - a2 * cos(2.0 * PI * i / NN);
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
void Blackman_harris(int N, float* win)
{
    int NN, i;
    float a1, a2;
    float a3, a4;

    a1 = 0.35875;
    a2 = 0.48829;
    a3 = 0.14128;
    a4 = 0.01168;

    NN = N - 1;
    for (i = 0; i < N; i++) {
        win[i] = a1 - a2 * cos(2.0 * PI * i / NN) + a3 * cos(4.0 * PI * i / NN) - a4 * cos(6.0 * PI * i / NN);
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
void hdn(float* hd, int N, float wc)
{
    int alpha;
    int i;
    float m;

    alpha = (N - 1) / 2;

    for (i = 0; i < N; i++) {
        m = (i - alpha + 1.0e-10);
        hd[i] = sin(wc * m) / (PI * m);
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

void hh(float* win, float* hd, float* h, int N)
{
    int i;

    for (i = 0; i < N; i++) {
        h[i] = hd[i] * win[i];
    }

    return;
}
/*====================================================================


/*==============================================================

  This subroutine is used for FFT/IFFT
   
===========================================================*/
void fft(float* xreal, float* ximag, int n, int sign)
{
    int i, j, k, m, temp;
    int h, q, p;
    float t;
    float *a, *b;
    float *at, *bt;
    int* r;

    //xreal=(float*)malloc(n*sizeof(float));
    //ximag=(float*)malloc(n*sizeof(float));
    a = (float*)malloc(n * sizeof(float));
    b = (float*)malloc(n * sizeof(float));
    r = (int*)malloc(n * sizeof(int));
    at = (float*)malloc(n * sizeof(float));
    bt = (float*)malloc(n * sizeof(float));

    m = (int)(log(n - 0.5) / log(2.0)) + 1; //2的幂，2的m次方等于n；
    for (i = 0; i < n; i++) {
        a[i] = xreal[i];
        b[i] = ximag[i];
        r[i] = i;
    }
    for (i = 0, j = 0; i < n - 1; i++) //0到n的反序；
    {
        if (i < j) {
            temp = r[i];
            r[i] = j;
            r[j] = temp;
        }
        k = n / 2;
        while (k < (j + 1)) {
            j = j - k;
            k = k / 2;
        }
        j = j + k;
    }

    t = 2 * PI / n;
    for (h = m - 1; h >= 0; h--) {
        p = (int)pow(2.0, h);
        q = n / p;
        for (k = 0; k < n; k++) {
            at[k] = a[k];
            bt[k] = b[k];
        }

        for (k = 0; k < n; k++) {
            if (k % p == k % (2 * p)) {

                a[k] = at[k] + at[k + p];
                b[k] = bt[k] + bt[k + p];
                a[k + p] = (at[k] - at[k + p]) * cos(t * (q / 2) * (k % p)) - (bt[k] - bt[k + p]) * sign * sin(t * (q / 2) * (k % p));
                b[k + p] = (bt[k] - bt[k + p]) * cos(t * (q / 2) * (k % p)) + (at[k] - at[k + p]) * sign * sin(t * (q / 2) * (k % p));
            }
        }
    }

    for (i = 0; i < n; i++) {
        if (sign == 1) {
            xreal[r[i]] = a[i];
            ximag[r[i]] = b[i];
        } else if (sign == -1) {
            xreal[r[i]] = a[i] / n;
            ximag[r[i]] = b[i] / n;
        }
    }

    free(a);
    free(b);
    free(r);
    free(at);
    free(bt);
}

/*=====================================================================

!	--------------------------------------------------------------
!   	This subroutine is used for calculate the con in time domain. 
!   	X(m) : input data
!   	H(n) : filter
!   	Y(l) : output
!   	2012.3.28/Jie Wang
!   	--------------------------------------------------------------

*=====================================================================*/

void con(float* x, int m, float* h, int n, float* out)
{
    float* y;
    int l, n_half;
    int i, k;
    l = m + n - 1;
    y = (float*)malloc(l * sizeof(float));
    for (k = 0; k < l; k++) {
        y[k] = 0.0;
        for (i = 0; i < m; i++) {
            if ((k - i) >= 0 && (k - i) <= (n - 1)) {
                y[k] = y[k] + x[i] * h[k - i];
            }
        }
    }

    n_half = n / 2;

    for (i = 0; i < m; i++) {
        out[i] = y[i + n_half];
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

void fre_filter(float* H, int NFFT, float w1, float w2, float df)
{
    int i;
    float f;

    for (i = 0; i < NFFT / 2 + 1; i++) {
        f = i * df;
        if (f <= w2 && f >= w1) {
            H[i] = 1.0;
        } else {
            H[i] = 0.0;
        }
    }

    for (i = NFFT / 2 + 1; i < NFFT; i++) {
        f = i * df;
        H[i] = H[NFFT + 1 - i]; // maybe has some problem...
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

void fre_filter_P(float* P, int NFFT, float f1, float f2, float df)
{
    float f;
    float delta_f, a, temp1, temp2, temp3;
    int it;

    delta_f = f2 - f1;
    a = 1.6;
    a = 2.0;

    for (it = 0; it < NFFT; it++) {
        f = it * df;
        if (f <= f1) {
            P[it] = 1.0;
        }
        if (f > f1 && f <= f2) {
            temp1 = f - f1;
            temp2 = delta_f / 2.0;
            temp3 = a * temp1 / temp2;
            P[it] = exp(-0.5 * temp3 * temp3);
        }
        if (f > f2) {
            P[it] = 0.0;
        }
    }

    return;
}
