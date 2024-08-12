# ncu profile 的使用指南

## 1. ncu 的安装与profile生成
Nsight Compute安装包在 [https://developer.nvidia.com/tools-overview/nsight-compute/get-started](https://developer.nvidia.com/tools-overview/nsight-compute/get-started) 可以获得。
![alt text](image.png)


可以通过`ncu --help`来查看ncu命令的参数
```
General Options:
  -h [ --help ]                         Print this help message.
  -v [ --version ]                      Print the version number.
  --mode arg (=launch-and-attach)       Select the mode of interaction with the target application:
                                          launch-and-attach
                                          (launch and attach for profiling)
                                          launch
                                          (launch and suspend for later attach)
                                          attach
                                          (attach to launched application)
  -p [ --port ] arg (=49152)            Base port for connecting to target application
  --max-connections arg (=64)           Maximum number of ports for connecting to target application
  --config-file arg (=1)                Use config.ncu-cfg config file to set parameters. Searches in the current 
                                        working directory and "$HOME/.config/NVIDIA Corporation" directory.
  --config-file-path arg                Override the default path for config file.

Launch Options:
  --check-exit-code arg (=1)            Check the application exit code and print an error if it is different than 0. 
                                        If set, --replay-mode application will stop after the first pass if the exit 
                                        code is not 0.
  --injection-path-32 arg (=../linux-desktop-glibc_2_11_3-x86)
                                        Override the default path for the 32-bit injection libraries.
  --injection-path-64 arg               Override the default path for the 64-bit injection libraries.
  --preload-library arg                 Prepend a shared library to be loaded by the application before the injection 
                                        libraries.
  --call-stack                          Enable CPU Call Stack collection.
  --nvtx                                Enable NVTX support.
  --support-32bit                       Support profiling processes launched from 32-bit applications.
  --target-processes arg (=all)         Select the processes you want to profile:
                                          application-only
                                          (profile only the application process)
                                          all
                                          (profile the application and its child processes)
  --target-processes-filter arg         Set the comma separated expressions to filter which processes are profiled.
                                          <process name> Set the exact process name to include for profiling.
                                          regex:<expression> Set the regex to include matching process names for 
                                        profiling.
                                            On shells that recognize regular expression symbols as special characters,
                                            the expression needs to be escaped with quotes.
                                          exclude:<process name> Set the exact process name to exclude for profiling.
                                          exclude-tree:<process name> Set the exact process name to exclude
                                            for profiling and further process tracking. None of its child processes
                                            will be profiled, even if they match a positive filter.
                                        The executable name part of the process will be considered in the match.
                                        Processing of filters stops at the first match.
                                        If any positive filter is specified, only processes matching a positive filter 
                                        are profiled.
  --null-stdin                          Launch the application with '/dev/null' as its standard input. This avoids 
                                        applications reading from standard input being stopped by SIGTTIN signals and 
                                        hanging when running as backgrounded processes.

Attach Options:
  --hostname arg                        Set hostname / ip address for connection target.

Common Profile Options:
 ...
```
在正常情况下，大多数参数并不需要使用，通常使用以下命令即可
```
ncu --set full -o *** python3 xxx.py
```
完成后会在服务器上产生一个 `***.ncu-rep` 文件, 可以在本地用 Nsight Compute 打开。
