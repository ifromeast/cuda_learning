import torch

a = torch.tensor([1., 2., 3.])

print(torch.square(a))
print(a ** 2)
print(a * a)

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

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

compiled_square = torch.compile(torch.square)

print(f"square time used: {time_pytorch_function(torch.square, b)} ms")
print(f"square 2 time used: {time_pytorch_function(square_2, b)} ms")
print(f"square 3 time used: {time_pytorch_function(square_3, b)} ms")
print(f"compiled square time used: {time_pytorch_function(compiled_square, b)} ms")

print("=============")
print("Profiling torch.square")
print("=============")

# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)
prof.export_chrome_trace("logs/square.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_2(b)
prof.export_chrome_trace("logs/square2.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_3(b)
prof.export_chrome_trace("logs/square3.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))