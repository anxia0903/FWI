void ricker_wave(float *rick, int itmax, float f0, float t0, float dt);

void get_acc_model(float *vp, float *vs, float *rho, int ntp, int ntx, int ntz);

void get_ini_model(float *vp, float *vs, float *rho, 
                   float *vp_n, float *vs_n,
                   int ntp, int ntx, int ntz);
void ini_model_mine(float *vp, float *vp_n, int ntp, int ntz, int ntx, int pml, int flag);

void maximum_vector(float *vector, int n, float *maximum_value);

void get_lame_constants(float *lambda, float *mu, 
     float *lambda_plus_two_mu, float *vp, 
     float * vs, float * rho, int ntp);
     
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
                      int pml, float dx, float f0, 
                      float t0, float dt, float *vp_max);


void fdtd_cpml_2d_GPU(int ntx, int ntz, int ntp, int nx, int nz,
                      int pml, float dx, float dz,
                      float *rick, int itmax, float dt, 
                      int is, int s_ix, int s_iz, float *rho, 
                      float *lambda, float *mu, 
                      float *lambda_plus_two_mu,
                      float *k_x, float *k_x_half,
                      float *k_z, float *k_z_half,
                      float *a_x, float *a_x_half,
                      float *a_z, float *a_z_half,
                      float *b_x, float *b_x_half,
                      float *b_z, float *b_z_half,
                      float *c, int inv_flag,
                      float *seismogram_vx,
                      float *seismogram_vz,
                      float *vx_borders_up, float *vx_borders_bottom,
                      float *vx_borders_left, float *vx_borders_right,
                      float *vz_borders_up, float *vz_borders_bottom,
                      float *vz_borders_left, float *vz_borders_right
                      );

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
    );

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
    );

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
    );

  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz(
    float *mu,
    float *a_x_half, float *a_z_half,
    float *b_x_half, float *b_z_half,
    float *vx, float *vz, float *sigmaxz,
    float *phi_vx_z, float *phi_vz_x,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt,
    int inv_flag
    );

  __global__ void fdtd_cpml_2d_GPU_kernel_vx_shared(
    float *rho,
    float *a_x_half, float *a_z, 
    float *b_x_half, float *b_z, 
    float *vx, float *sigmaxx, float *sigmaxz,
    float *phi_sigmaxx_x, float *phi_sigmaxz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt,
    float *snap
    );

  __global__ void fdtd_cpml_2d_GPU_kernel_vz_shared(
    float *rho,
    float *a_x, float *a_z_half,
    float *b_x, float *b_z_half,
    float *vz, float *sigmaxz, float *sigmazz, 
    float *phi_sigmaxz_x, float *phi_sigmazz_z,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    );


  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_shared(
    float *rick, 
    float *lambda, float *lambda_plus_two_mu,
    float *a_x, float *a_z,
    float *b_x, float *b_z,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    float *phi_vx_x, float *phi_vz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt, int s_ix, int s_iz, int it
    );

  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_shared(
    float *mu,
    float *a_x_half, float *a_z_half,
    float *b_x_half, float *b_z_half,
    float *vx, float *vz, float *sigmaxz,
    float *phi_vx_z, float *phi_vz_x,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    );

  __global__ void fdtd_2d_GPU_kernel_vx(
    float *rho,
    float *vx, float *sigmaxx, float *sigmaxz,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    );

  __global__ void fdtd_2d_GPU_kernel_vz(
    float *rho,
    float *vz, float *sigmaxz, float *sigmazz,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    );

  __global__ void fdtd_2d_GPU_kernel_sigmaxx_sigmazz(
    float *rick, 
    float *lambda, float *lambda_plus_two_mu,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt, int s_ix, int s_iz, int it
    );

  __global__ void fdtd_2d_GPU_kernel_sigmaxz(
    float *mu,
    float *vx, float *vz, float *sigmaxz,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    );


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
                    float *seismogram_vx_rms,
                    float *seismogram_vz_rms,
                    float *vx_borders_up, float *vx_borders_bottom,
                    float *vx_borders_left, float *vx_borders_right,
                    float *vz_borders_up, float *vz_borders_bottom,
                    float *vz_borders_left, float *vz_borders_right,
                    float *image_lambda, float *image_mu,
                    float *image_sources, float *image_receivers);

  __global__ void fdtd_2d_GPU_kernel_sigmaxx_sigmazz_backward(
    float *rick, 
    float *lambda, float *lambda_plus_two_mu,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    int ntp, int ntx, int ntz, int pml,
    float dx, float dz, float dt, int s_ix, int s_iz, int it
    );

  __global__ void fdtd_2d_GPU_kernel_sigmaxz_backward(
    float *mu,
    float *vx, float *vz, float *sigmaxz,
    int ntp, int ntx, int ntz, int pml,
    float dx, float dz, float dt
    );

  __global__ void fdtd_2d_GPU_kernel_vx_backward(
    float *rho,
    float *vx, float *sigmaxx, float *sigmaxz,
    int ntp, int ntx, int ntz, int pml,
    float dx, float dz, float dt
    );

  __global__ void fdtd_2d_GPU_kernel_vz_backward(
    float *rho,
    float *vz, float *sigmaxz, float *sigmazz,
    int ntp, int ntx, int ntz, int pml,
    float dx, float dz, float dt
    );

  __global__ void fdtd_2d_GPU_kernel_borders_backward
  (
   float *vx,
   float *vx_borders_up, float *vx_borders_bottom,
   float *vx_borders_left, float *vx_borders_right,
   float *vz,
   float *vz_borders_up, float *vz_borders_bottom,
   float *vz_borders_left, float *vz_borders_right,
   int ntp, int ntx, int ntz, int pml, int it
  );

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
    );

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
    );

  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_backward(
    float *lambda, float *lambda_plus_two_mu,
    float *a_x, float *a_z,
    float *b_x, float *b_z,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    float *phi_vx_x, float *phi_vz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    );

  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_backward(
    float *mu,
    float *a_x_half, float *a_z_half,
    float *b_x_half, float *b_z_half,
    float *vx, float *vz, float *sigmaxz,
    float *phi_vx_z, float *phi_vz_x,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    );

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
    );


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
    );

  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxx_sigmazz_mine_backward(
    float *a_x, float *a_z,
    float *b_x, float *b_z,
    float *vx, float *vz, float *sigmaxx, float *sigmazz,
    float *phi_vx_x, float *phi_vz_z, 
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    );

  __global__ void fdtd_cpml_2d_GPU_kernel_sigmaxz_mine_backward(
    float *a_x_half, float *a_z_half,
    float *b_x_half, float *b_z_half,
    float *vx, float *vz, float *sigmaxz,
    float *phi_vx_z, float *phi_vz_x,
    int ntp, int ntx, int ntz, 
    float dx, float dz, float dt
    );

  __global__ void sum_image_GPU_kernel_lambda
  (
    float *vx_inv, float *vz_inv,
    float *vx, float *vz, 
    float *sigmaxx, float *sigmazz, 
    float *image_lambda, float *image_sources, float *image_receivers,
    int ntx, int ntz, int pml, float dx, float  dz
  );

  __global__ void sum_image_GPU_kernel_mu
  (
    float *vx_inv, float *vz_inv, 
    float *sigmaxx, float *sigmazz, float *sigmaxz,
    float *image, 
    int ntx, int ntz, int pml, float dx, float dz
  );

  __global__ void sum_image_GPU_kernel_sources
  (
    float *vx_inv, float *vz_inv, 
    float *image, 
    int ntx, int ntz, int pml, float dx, float dz
  );

  __global__ void sum_image_GPU_kernel_receivers
  (
    float *sigmaxx, float *sigmazz, float *image, 
    int ntx, int ntz, int pml, float dx, float dz
  );

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
       float *dn_vp_pre, float *dn_vs_pre);

void ini_step(float *dn, int np, float *un0);

void update_model(float *vp, float *vs, float *rho, 
                  float *vp_n, float *vs_n,
                  float *dn_vp, float *un_vp, float *dn_vs, float *un_vs,
                  int ntp, int ntz, int ntx, int pml);
void Preprocess(int nz, int nx, float dx, float dz, float *P);
void Preprocess_vs(int nz, int nx, float dx, float dz, float *P);
