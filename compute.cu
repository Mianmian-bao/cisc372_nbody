#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void compute_accels(vector3* accels, vector3* pos, double* m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n) return;

    int idx = i * n + j;

    if (i == j) {
        accels[idx][0] = 0;
        accels[idx][1] = 0;
        accels[idx][2] = 0;
    } else {
        vector3 distance;

        for (int k = 0; k < 3; k++) {
            distance[k] = pos[i][k] - pos[j][k];
        }

        double magnitude_sq =
            distance[0] * distance[0] +
            distance[1] * distance[1] +
            distance[2] * distance[2];

        double magnitude = sqrt(magnitude_sq);
        double accelmag = -1 * GRAV_CONSTANT * m[j] / magnitude_sq;

        accels[idx][0] = accelmag * distance[0] / magnitude;
        accels[idx][1] = accelmag * distance[1] / magnitude;
        accels[idx][2] = accelmag * distance[2] / magnitude;
    }
}

__global__ void sum_and_update(vector3* accels, vector3* pos, vector3* vel, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    vector3 accel_sum = {0, 0, 0};

    for (int j = 0; j < n; j++) {
        for (int k = 0; k < 3; k++) {
            accel_sum[k] += accels[i * n + j][k];
        }
    }

    for (int k = 0; k < 3; k++) {
        vel[i][k] += accel_sum[k] * INTERVAL;
        pos[i][k] += vel[i][k] * INTERVAL;
    }
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
extern "C" void compute() {
    int n = NUMENTITIES;

    vector3 *d_pos, *d_vel, *d_accels;
    double *d_mass;

    cudaMalloc(&d_pos, sizeof(vector3) * n);
    cudaMalloc(&d_vel, sizeof(vector3) * n);
    cudaMalloc(&d_mass, sizeof(double) * n);
    cudaMalloc(&d_accels, sizeof(vector3) * n * n);

    cudaMemcpy(d_pos, hPos, sizeof(vector3) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, hVel, sizeof(vector3) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * n, cudaMemcpyHostToDevice);

    dim3 block2D(16, 16);
    dim3 grid2D((n + block2D.x - 1) / block2D.x,
                (n + block2D.y - 1) / block2D.y);

    compute_accels<<<grid2D, block2D>>>(d_accels, d_pos, d_mass, n);
    cudaDeviceSynchronize();

    int block1D = 256;
    int grid1D = (n + block1D - 1) / block1D;

    sum_and_update<<<grid1D, block1D>>>(d_accels, d_pos, d_vel, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hPos, d_pos, sizeof(vector3) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_vel, sizeof(vector3) * n, cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_mass);
    cudaFree(d_accels);
}