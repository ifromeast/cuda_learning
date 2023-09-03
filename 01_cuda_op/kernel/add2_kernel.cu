__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

__global__ void MatAdd(float* c,
                            const float* a,
                            const float* b,
                            int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j*n + i;
    if (i < n && j < n)
        c[idx] = a[idx] + b[idx]
}

void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 block(16, 16);
    dim3 grid(n/block.x, n/block.y);

    MatAdd<<<grid, block>>>(c, a, b, n);
}