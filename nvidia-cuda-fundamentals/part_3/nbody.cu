#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"

#define SOFTENING 1e-9f

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */

void bodyForce(Body *p, float dt, int n) {
  for (int i = 0; i < n; ++i) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

__global__
void cuBodyForce(Body *p, float dt, int n) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;
    
    for (; i < n; i += gridStride)
    {
       if (i < n){
           float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

            for (int j = 0; j < n; j++) {
              float dx = p[j].x - p[i].x;
              float dy = p[j].y - p[i].y;
              float dz = p[j].z - p[i].z;
              float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
              float invDist = rsqrtf(distSqr);
              float invDist3 = invDist * invDist * invDist;

              Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
            }

            p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
       }
    }
}

__global__
void cuIntegrate(Body *p, float dt, int nBodies) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;
    
    for (; i < nBodies; i += gridStride)
    {
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }
}

int main(const int argc, const char** argv) {
  /* get device properties */
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  
  printf("Device ID: %d\tNumber of SMs: %d\n", deviceId, numberOfSMs);
  
  size_t threadsPerBlock = 256;
  size_t numberOfBlocks = 32 * numberOfSMs;
  
  // The assessment will test against both 2<11 and 2<15.
  // Feel free to pass the command line argument 15 when you generate ./nbody report files
  int nBodies = 2<<11;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  // The assessment will pass hidden initialized values to check for correctness.
  // You should not make changes to these files, or else the assessment will not work.
  const char * initialized_values;
  const char * solution_values;

  if (nBodies == 2<<11) {
    initialized_values = "09-nbody/files/initialized_4096";
    solution_values = "09-nbody/files/solution_4096";
  } else { // nBodies == 2<<15
    initialized_values = "09-nbody/files/initialized_65536";
    solution_values = "09-nbody/files/solution_65536";
  }

  if (argc > 2) initialized_values = argv[2];
  if (argc > 3) solution_values = argv[3];

  const float dt = 0.01f; // Time step
  const int nIters = 10;  // Simulation iterations

  // initialize p
  int bytes = nBodies * sizeof(Body);
  float *buf;
  cudaMallocManaged(&buf, bytes);
  Body *p = (Body*)buf;
  
  // add gpu pre-fetching logic here
  cudaMemPrefetchAsync(p, bytes, deviceId);

  read_values_from_file(initialized_values, buf, bytes);
  double totalTime = 0.0;

  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();

    cuBodyForce<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies); // compute interbody forces
    cudaDeviceSynchronize();

    cuIntegrate<<<numberOfBlocks, threadsPerBlock>>>(p, dt, nBodies); // compute interbody forces
    cudaDeviceSynchronize();

    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
  write_values_to_file(solution_values, buf, bytes);

  // You will likely enjoy watching this value grow as you accelerate the application,
  // but beware that a failure to correctly synchronize the device might result in
  // unrealistically high values.
  printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

  free(buf);
}
