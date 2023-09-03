__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

// Kernel definition
__global__ void MatAdd(float *c,
                       const float *a,
                       const float *b,
                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n)
        c[i] = a[i] + b[i];
}



void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 grid(16,16);
    dim3 block(n/grid.x, n/grid.y);
    MatAdd<<<grid, block>>>(c, a, b, n);
}