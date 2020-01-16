/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

#define PROJECT_DEF 1

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

/*The algorithm first computes the real and imaginary components of mu 
at each sample point in the trajectory space (k-space), 
then computes the real and imaginarycomponents of FHd at each voxel in the image space.*/


#if PROJECT_DEF
#define BLOCK_SIZE 512
#define K_VALS_GRID_SIZE (BLOCK_SIZE * 4)
__constant__ __device__ kValues const_kValues[K_VALS_GRID_SIZE];

//calculate mu at each sample point t
__global__ void ComputePhiMagKernel(int numK, float *phiR, float *phiI,
                                    float *phiMag)
{
	
// find the index of the voxel assigned to this thread
  unsigned int t = threadIdx.x + (blockIdx.x * blockDim.x);
  if (t < numK)
    phiMag[t] = (phiR[t] * phiR[t]) + (phiI[t] * phiI[t]);
}
/*The GPU-based implementation of the FHd algorithm uses constant memory caches to eliminate the
potential bottleneck posed by memory bandwidth and latency. 

The scan data is divided into many tiles, the host CPU loads the corresponding subset of sample points into constant memory 
before executing the cmpFhD function. 

Each thread computes a partial sum for a single element of FHd by iterating over all the sample points in the tile.

This optimization significantly increases the ratio of FP operations to global memory accesses.*/
//calculate FHd on one voxel//
__global__ void ComputeQKernel(int numK, int numX,
                               float *x_d, float *y_d, float *z_d,
                               float *Qr_d, float *Qi_d)
{ 
// find the index of the voxel assigned to this thread
  unsigned int t = threadIdx.x + (blockIdx.x * blockDim.x);

  if (t >= numX)
    return;
//register allocate voxel inputs and outputs
  float x_l = x_d[t];
  float y_l = y_d[t];
  float z_l = z_d[t];
  float Qracc = 0.0f;
  float Qiacc = 0.0f;
  float phi = 0.0f;

  float expArg;
  int idx = 0;
	

  if (numK % 2) {	  
    /* if numK is odd */
	// e^2pi*km*xn  
    expArg = PIx2 * (const_kValues[idx].Kx * x_l +
                     const_kValues[idx].Ky * y_l +
                     const_kValues[idx].Kz * z_l);
    phi = const_kValues[idx].PhiMag;
    /*First, instead of fath math, changing cos() and sin() function to hardware versions: __sin() and __cos(). 
     Because CUDA offers hardware implementations of mathematic functions that 
     provide much higher throughput than their software counterparts, but it will 
     reduced accuracy when switching from software functions to hardware functions.
     So must carefully.*/  
    /*failed to use five element taylor */
    Qracc += phi * _cos(expArg);
    Qiacc += phi * _sin(expArg);
    idx++;
  }

  for (; idx < numK; idx++) {
    /* using thread coarsening technique */
   //const_kValues(sample data)is held in costant memory
    expArg = PIx2 * (const_kValues[idx].Kx * x_l +
                     const_kValues[idx].Ky * y_l +
                     const_kValues[idx].Kz * z_l);

    phi = const_kValues[idx].PhiMag;
    Qracc += phi * _cos(expArg);
    Qiacc += phi * _sin(expArg);

    idx++;
    expArg = PIx2 * (const_kValues[idx].Kx * x_l +
                     const_kValues[idx].Ky * y_l +
                     const_kValues[idx].Kz * z_l);

    phi = const_kValues[idx].PhiMag;
   /*hardware versions: __sin() and __cos()*/
    Qracc += phi * _cos(expArg);
    Qiacc += phi * _sin(expArg);
  }
  Qr_d[t] += Qracc;
  Qi_d[t] += Qiacc;
}

void ComputePhiMagGPU(int numK, float* phiR_d, float* phiI_d,
                      float* phiMag_d)
{
  unsigned int numBlocks = ((numK - 1) / BLOCK_SIZE) + 1;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  ComputePhiMagKernel<<<dimGrid, dimBlock>>>(numK, phiR_d, phiI_d, phiMag_d);
}

void ComputeQGPU(int numK, int numX, struct kValues *kVals,
                 float *x_d, float *y_d, float *z_d, float *Qr_d, float *Qi_d)
{
  unsigned int size_to_cover = K_VALS_GRID_SIZE;
  unsigned int n_iter = ((numK - 1) / K_VALS_GRID_SIZE) + 1;
  struct kValues *ptr = kVals;

  unsigned int numBlocks = ((numX - 1) / BLOCK_SIZE) + 1;
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  //printf("size : %d\n", sizeof(struct kValues));

  for (int iter = 0; iter < n_iter; iter++) {
    size_to_cover = MIN(K_VALS_GRID_SIZE, numK - (iter * K_VALS_GRID_SIZE));
    //printf("size to cover:%d, iter:%d, ptr:%u\n", size_to_cover, iter, ptr);
    if (size_to_cover) {
        cudaMemcpyToSymbol(const_kValues, ptr, size_to_cover * sizeof(struct kValues), 0);
        ComputeQKernel<<<dimGrid, dimBlock>>>(size_to_cover, numX, x_d, y_d, z_d, Qr_d, Qi_d);
        if (cudaSuccess != cudaDeviceSynchronize()) {
            printf("iter: %d ERROR!!!!!!\n", iter);
        }
    }
    ptr += size_to_cover;
  }

}

#else

inline
void
ComputePhiMagCPU(int numK,
                 float* phiR, float* phiI,
                 float* __restrict__ phiMag) {
  int indexK = 0;
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

inline
void
ComputeQCPU(int numK, int numX,
            struct kValues *kVals,
            float* x, float* y, float* z,
            float *__restrict__ Qr, float *__restrict__ Qi) {
  float expArg;
  float cosArg;
  float sinArg;

  int indexK, indexX;

  // Loop over the space and frequency domains.
  // Generally, numX > numK.
  // Since loops are not tiled, it's better that the loop with the smaller
  // cache footprint be innermost.
  for (indexX = 0; indexX < numX; indexX++) {

    // Sum the contributions to this point over all frequencies
    float Qracc = 0.0f;
    float Qiacc = 0.0f;
    for (indexK = 0; indexK < numK; indexK++) {
      expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
                       kVals[indexK].Ky * y[indexX] +
                       kVals[indexK].Kz * z[indexX]);

      cosArg = cosf(expArg);
      sinArg = sinf(expArg);

      float phi = kVals[indexK].PhiMag;
      Qracc += phi * cosArg;
      Qiacc += phi * sinArg;
    }
    Qr[indexX] = Qracc;
    Qi[indexX] = Qiacc;
  }
}

#endif

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}
