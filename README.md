# 🤗 Transformers Benchmarking

Benchmark the performance of 🤗 Transformers on PyTorch. Quick project done out of curiosity. 

Live link: https://share.streamlit.io/tlkh/transformers-benchmarking/main/app.py

```shell
docker pull nvcr.io/nvidia/pytorch:20.10-py3

# inside container
pip install -r requirements.txt -U

# edit EXP_PREFIX to change directory to record results
nano run_benchmarks_1gpu.sh

# run benchmarking script
run_benchmarks_1gpu.sh
```

### More Information

**Important disclaimer**

Benchmarks are done by myself independently on with NVIDIA NGC PyTorch 20.10 container. The information is provided on a best-effort basis and I make no guarantees of the accuracy of the information. Feel free to [open a GitHub issue](https://github.com/tlkh/transformers-benchmarking/issues/new) if you have any questions or suggestions.

This dashboard shows the measured performance of GPUs when training various configurations of Transformer networks, showing throughput (seq/s) and GPU memory (VRAM) usage. The idea is to allow users have an easy reference for choosing model configuration (model size/batch size/sequence length) and GPU model.

Each HuggingFace Transformer model is loaded in PyTorch as a `AutoModelForSequenceClassification` and used in a PyTorch Lightning `LightningModule` for training.

Forward and backward pass computation timings and provided by the PyTorch Lightning built-in [`SimpleProfiler`](https://pytorch-lightning.readthedocs.io/en/latest/profiler.html#enable-simple-profiling).

GPU utilization shown is likely lower than actual since measuring GPU utilization (especially using [DLProf](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/) to measure Tensor Core utilization) likely reduces GPU utilization due to overheads. This is not present in the throughput measurements.

All current results are using PyTorch [native automatic mixed precision](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) and [Apex fused AdamW optimizer](https://nvidia.github.io/apex/optimizers.html).

Systems used:
* V100 - local OEM system
* A100 - AWS p4d.24xlarge instance

Only single GPU configurations have been tested.
