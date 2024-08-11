# 初识典型框架语言的使用方法及分析手段

本节将通过一个简单的 element-wise 的例子（矩阵平方），来展示不同框架如何进行GPU编程，并分析其性能。

## 1. PyTorch 的实现与profile
- 代码：pytorch_square.py
- 命令：python pytorch_square.py

PyTorch 可通过三种方式实现矩阵平方操作：内置函数、pow 函数、mul 函数。
```
torch.square(a)
def square_2(a):
    return a * a
def square_3(a):
    return a ** 2
```
使用`cuda.Event`记录执行时间
```
def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

b = torch.randn(10000, 10000).cuda()
print(f"square time used: {time_pytorch_function(torch.square, b)} ms")
print(f"square 2 time used: {time_pytorch_function(square_2, b)} ms")
print(f"square 3 time used: {time_pytorch_function(square_3, b)} ms")
```
在 H100 中结果如下所示：
```
square time used: 0.26956799626350403 ms
square 2 time used: 0.2685759961605072 ms
square 3 time used: 0.2667199969291687 ms
```
接下来生成profile文件,以内置函数为例
```
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)
prof.export_chrome_trace("logs/square.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```
可以在`chrome://tracing/`中加载查看profile文件，如下：
![alt text](img/image.png)
打印出来的结果如下所示
```
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                aten::mul         8.36%      23.000us        12.73%      35.000us      35.000us     298.000us       100.00%     298.000us     298.000us             1  
          cudaEventRecord         2.91%       8.000us         2.91%       8.000us       4.000us       0.000us         0.00%       0.000us       0.000us             2  
         cudaLaunchKernel         4.36%      12.000us         4.36%      12.000us      12.000us       0.000us         0.00%       0.000us       0.000us             1  
    cudaDeviceSynchronize        84.36%     232.000us        84.36%     232.000us     232.000us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 275.000us
Self CUDA time total: 298.000us
```
如果希望看到更多信息可以使用（pt_profiler.py）
```
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        a = torch.square(torch.randn(10000, 10000).cuda())

prof.export_chrome_trace("logs/trace.json")
```
可以清楚看到，计算重复了10次，且第一次warmup的时间明显更长
![alt text](img/image-1.png)
展开其中一条记录，可以看到所用的核函数及其所用时间
![alt text](img/image-2.png)

