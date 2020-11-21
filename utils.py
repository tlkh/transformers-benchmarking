import os
import ctypes
import subprocess
import json
import torch
from torch.utils import collect_env

def increase_l2_fetch_granularity():
    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128

def get_gpu_memory_info():
    """Records GPU used and total memory info in GB"""
    _output_to_list = lambda x: x.decode("ascii").split("\n")[:-1]
    info = {
        "total": None,
        "used": None,
    }
    for i in info.keys():
        COMMAND = "nvidia-smi --query-gpu=memory."+i+" --format=csv"
        memory_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
        memory_values = [int(x.split()[0])/1000 for i, x in enumerate(memory_info)]
        info[i] = memory_values
    return info
        
def get_gpu_hw_info(output_file="./gpu_hw_info.txt"):
    COMMAND = "./deviceQuery"
    info = subprocess.check_output(COMMAND.split()).decode("ascii")
    with open(output_file, "w") as writer:
        writer.write(info)

def enable_tokenizers_parallelism():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
def enable_cuda_cache():
    os.environ["CUDA_CACHE_DISABLE"] = "0"

def collect_env_info(output_file="./env_info.txt"):
    env_info = collect_env.get_pretty_env_info()
    with open(output_file, "w") as f:
        f.write(env_info) 
        
def dump_json(data, output_file="./exp_info.json"):
    with open(output_file, "w") as outfile:
        json.dump(data, outfile)
        
def setup_env(profiling=True, nvidia=True, output_file="./env_info.txt"):
    torch.autograd.set_detect_anomaly(False)
    enable_tokenizers_parallelism()
    if profiling:
        print("Profiling enabled: torch.autograd.profiler.profile(True)")
        torch.autograd.profiler.profile(True)
    else:
        torch.autograd.profiler.profile(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    if nvidia:
        enable_cuda_cache()
        increase_l2_fetch_granularity()
        if profiling:
            print("Profiling enabled: torch.autograd.profiler.emit_nvtx(True)")
            torch.autograd.profiler.emit_nvtx(True)
        else:
            torch.autograd.profiler.emit_nvtx(False)
    collect_env_info(output_file=output_file)
        