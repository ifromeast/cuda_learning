# 初识各种各种框架语言的使用方法及分析手段

本节将通过一个简单的 element-wise 的例子，来展示不同框架如何进行GPU编程，并分析其性能。

## 1. CUDA 代码

```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}