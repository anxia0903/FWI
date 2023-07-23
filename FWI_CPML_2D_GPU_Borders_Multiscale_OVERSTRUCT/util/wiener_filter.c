
/*******************************************************************************

         --------------------------------------------------------------
        ! This program is used for testing the WIENER filter.
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
        ! 2013.4.7/Jie Wang
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

#define EPS 1.0e-20

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

void fft(float* xreal, float* ximag, int n, int sign);
void ricker_wave(float* rick, int itmax, float f0, float t0, float dt);

int main()
{
    int slen, it, ix, ip;

    FILE* fp;
    float* data;
    /*=========================================================
  Parameters of ricker wave...
  ========================================================*/

    int itmax = 1500;
    float* rick;
    float f0 = 20.0f;
    float t0 = 0.20f;
    float dt = 1.00e-3f;

    float rick_amp;

    float* rick_target;
    float f0_target = 5.0;
    float t0_target = t0;

    //    float *filter_wiener;
    float* rick_filted;

    float* rick_fft_real;
    float* rick_fft_imag;

    float* rick_target_fft_real;
    float* rick_target_fft_imag;

    float* filter_wiener_fft_real;
    float* filter_wiener_fft_imag;

    float* rick_filted_fft_real;
    float* rick_filted_fft_imag;

    Complex* out;

    int nx = 384;

    char filename[30];

    float* seismogram_vx_obs;
    float** seis_vx_obs;
    float** seis_vx_obs_filted;

    seismogram_vx_obs = (float*)malloc(sizeof(float) * itmax * nx);

    seis_vx_obs = (float**)malloc(sizeof(float*) * itmax);
    seis_vx_obs_filted = (float**)malloc(sizeof(float*) * itmax);

    for (it = 0; it < itmax; it++) {
        seis_vx_obs[it] = (float*)malloc(sizeof(float) * nx);
        seis_vx_obs_filted[it] = (float*)malloc(sizeof(float) * nx);
    }

    /*=========================================================
  Parameters of the filter...
  ========================================================*/

    int dimension_flag = 1;
    int K, NFFT;
    float df, f_total;

    /*=========================================================
  Calculate the number of fft points...
  ========================================================*/

    K = ceil(log(1.0 * itmax) / log(2.0));
    NFFT = pow(2, K);

    //------------Calculate the bandwidth of the filter ---------------

    df = 1.0 / (NFFT * dt);
    f_total = (NFFT - 1) * df; //the maxum of frequency

    //------------Allocate memory to vectors---------------------------

    rick = (float*)malloc(sizeof(float) * itmax);
    rick_target = (float*)malloc(sizeof(float) * itmax);
    rick_filted = (float*)malloc(sizeof(float) * itmax);

    rick_fft_real = (float*)malloc(sizeof(float) * NFFT);
    rick_fft_imag = (float*)malloc(sizeof(float) * NFFT);

    filter_wiener_fft_real = (float*)malloc(sizeof(float) * NFFT);
    filter_wiener_fft_imag = (float*)malloc(sizeof(float) * NFFT);

    rick_target_fft_real = (float*)malloc(sizeof(float) * NFFT);
    rick_target_fft_imag = (float*)malloc(sizeof(float) * NFFT);

    rick_filted_fft_real = (float*)malloc(sizeof(float) * NFFT);
    rick_filted_fft_imag = (float*)malloc(sizeof(float) * NFFT);
    out = (Complex*)malloc(sizeof(Complex) * NFFT);

    //    rick_new=(float*)malloc(sizeof(float)*NFFT);
    //    data_out=(float*)malloc(sizeof(float)*NFFT);

    /*##################################################################
  -------------------   The Part Of Filter   -----------------------
  ################################################################*/

    /*#####################################################################
//-----Calculate the ricker wavelet/input the data that needed to --
//-----be filtered -------------------------------------------------
/*####################################################################*/
    if (dimension_flag == 1) {
        //------Calculate the information of the original ricker wave-------
        ricker_wave(rick, itmax, f0, t0, dt);

        fp = fopen("rick_wave.dat", "w");
        for (it = 0; it < itmax; it++) {
            fprintf(fp, "%f\n", rick[it]);
        }
        fclose(fp);

        for (it = 0; it < NFFT; it++) {
            rick_fft_real[it] = 0.0;
            rick_fft_imag[it] = 0.0;
        }

        for (it = 0; it < itmax; it++) {
            rick_fft_real[it] = rick[it];
        }

        fft(rick_fft_real, rick_fft_imag, NFFT, 1);

        fp = fopen("rick_wave_amp.dat", "w");
        for (it = 0; it < 100; it++) {
            //fprintf(fp,"%18f%18f\n",it*df,sqrt(pow(rick_fft_imag[it],2)+pow(rick_fft_real[it],2)));
            fprintf(fp, "%f\n", sqrt(pow(rick_fft_imag[it], 2) + pow(rick_fft_real[it], 2)));
        }
        fclose(fp);

        //------Calculate the information of the target ricker wave-------
        ricker_wave(rick_target, itmax, f0_target, t0_target, dt);

        fp = fopen("rick_wave_target.dat", "w");
        for (it = 0; it < itmax; it++) {
            fprintf(fp, "%f\n", rick_target[it]);
        }
        fclose(fp);

        for (it = 0; it < NFFT; it++) {
            rick_target_fft_real[it] = 0.0;
            rick_target_fft_imag[it] = 0.0;
        }

        for (it = 0; it < itmax; it++) {
            rick_target_fft_real[it] = rick_target[it];
        }

        fft(rick_target_fft_real, rick_target_fft_imag, NFFT, 1);

        fp = fopen("rick_wave_target_amp.dat", "w");
        for (it = 0; it < 100; it++) {
            fprintf(fp, "%f\n", sqrt(pow(rick_target_fft_imag[it], 2) + pow(rick_target_fft_real[it], 2)));
        }
        fclose(fp);

        //---Calculate the Wiener filter.....
        for (it = 0; it < NFFT; it++) {
            rick_amp = pow(rick_fft_real[it], 2) + pow(rick_fft_imag[it], 2) + EPS;

            filter_wiener_fft_real[it] = (rick_fft_real[it] * rick_target_fft_real[it] + rick_fft_imag[it] * rick_target_fft_imag[it]) / rick_amp;
            filter_wiener_fft_imag[it] = (rick_fft_real[it] * rick_target_fft_imag[it] - rick_fft_imag[it] * rick_target_fft_real[it]) / rick_amp;
        }

        //----Filte the original ricker wave...

        for (it = 0; it < NFFT; it++) {
            rick_filted_fft_real[it] = rick_fft_real[it] * filter_wiener_fft_real[it] + rick_fft_imag[it] * filter_wiener_fft_imag[it];
            rick_filted_fft_imag[it] = rick_fft_real[it] * filter_wiener_fft_imag[it] + rick_fft_imag[it] * filter_wiener_fft_real[it];
        }

        fp = fopen("rick_wave_filted_amp.dat", "w");
        for (it = 0; it < 100; it++) {
            fprintf(fp, "%f\n", sqrt(pow(rick_filted_fft_imag[it], 2) + pow(rick_filted_fft_real[it], 2)));
        }
        fclose(fp);

        fft(rick_filted_fft_real, rick_filted_fft_imag, NFFT, -1);

        fp = fopen("rick_wave_filted.dat", "w");
        for (it = 0; it < itmax; it++) {
            rick_filted[it] = rick_filted_fft_real[it];
            fprintf(fp, "%f\n", rick_filted[it]);
        }
        fclose(fp);

        sprintf(filename, "./output/%dsource_seismogram_vx_obs.dat", 1);
        fp = fopen(filename, "rb");
        fread(&seismogram_vx_obs[0], sizeof(float), nx * itmax, fp);
        fclose(fp);

        for (it = 0; it < itmax; it++) {
            for (ix = 0; ix < nx; ix++) {
                ip = it * nx + ix;
                seis_vx_obs[it][ix] = seismogram_vx_obs[ip];
            }
        }

        for (ix = 0; ix < nx; ix++) {

            for (it = 0; it < NFFT; it++) {
                rick_fft_real[it] = 0.0;
                rick_fft_imag[it] = 0.0;
            }

            for (it = 0; it < itmax; it++) {
                rick_fft_real[it] = seis_vx_obs[it][ix];
            }

            fft(rick_fft_real, rick_fft_imag, NFFT, 1);

            for (it = 0; it < NFFT; it++) {
                rick_filted_fft_real[it] = rick_fft_real[it] * filter_wiener_fft_real[it] + rick_fft_imag[it] * filter_wiener_fft_imag[it];
                rick_filted_fft_imag[it] = rick_fft_real[it] * filter_wiener_fft_imag[it] + rick_fft_imag[it] * filter_wiener_fft_real[it];
            }
            fft(rick_filted_fft_real, rick_filted_fft_imag, NFFT, -1);

            for (it = 0; it < itmax; it++) {
                seis_vx_obs_filted[it][ix] = rick_filted_fft_real[it];
            }
        }

        for (it = 0; it < itmax; it++) {
            for (ix = 0; ix < nx; ix++) {
                ip = it * nx + ix;
                seismogram_vx_obs[ip] = seis_vx_obs_filted[it][ix];
            }
        }

        sprintf(filename, "./output/%dsource_seismogram_vx_obs.dat", 11);
        fp = fopen(filename, "wb");
        fwrite(&seismogram_vx_obs[0], sizeof(float), nx * itmax, fp);
        fclose(fp);
    }

    free(rick);
    free(rick_target);
    free(rick_filted);

    free(rick_fft_real);
    free(rick_fft_imag);

    free(filter_wiener_fft_real);
    free(filter_wiener_fft_imag);

    free(rick_target_fft_real);
    free(rick_target_fft_imag);

    free(rick_filted_fft_real);
    free(rick_filted_fft_imag);
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
