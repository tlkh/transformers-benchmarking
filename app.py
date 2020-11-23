import streamlit as st
st.set_page_config(
    page_title="Transformers Benchmarks",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded",
)
import pandas as pd
import plotly.express as px

SEQ_LEN_LIST = [128, 256, 512]
MODEL_LIST = [
    "distilroberta-base",
    "roberta-base",
    "roberta-large"
]
GPU_LIST = ["v100-16gb-300w", "a100-40gb-400w"]
COMPUTE_LIST = ["amp"]

st.sidebar.markdown("# 🤗 Transformers Benchmarking")
model_name = st.sidebar.selectbox("Model Name", MODEL_LIST)
gpu_name = st.sidebar.selectbox("GPU", GPU_LIST)
seq_len = st.sidebar.selectbox("Sequence Length", SEQ_LEN_LIST)
with st.sidebar.beta_expander("Additional Options", expanded=True):
    st.markdown("Ignore time between training steps since it can differ depending on CPU/dataloader")
    ignore_cpu = st.checkbox("Ignore CPU overhead", value=False)
    st.markdown("Show inference timing and latency (*estimated* from forward pass)")
    show_infer = st.checkbox("Show inference performance", value=False)
    st.markdown("Currently all results use automatic mixed precision")
    compute_name = st.selectbox("Compute Type", COMPUTE_LIST)
    st.markdown("For more information, please see **More information** section below the graphs")

df = pd.read_csv("./results.csv")
df_sel = df[df["seq_len"] == seq_len]
df_sel = df_sel[df["gpu"] == gpu_name]
df_sel = df_sel[df["model"] == model_name]
df_sel = df_sel[df["compute"] == compute_name]
df_sel.sort_values(by="batch_size", inplace=True)

def sum_and_reciprocal_to_col(frame, new_col_name, list_of_cols_to_sum):
    frame[new_col_name] = frame[list_of_cols_to_sum].astype(float).sum(axis=1)
    frame[new_col_name] = frame["batch_size"] * (1/frame[new_col_name])
    return(frame)

if ignore_cpu:
    compute_segments = ["forward", "backward"]
else:
    compute_segments = ["cpu_time", "forward", "backward"]

df_sel = sum_and_reciprocal_to_col(df_sel, "throughput", compute_segments)
df_sel["vram_usage"] = df_sel["vram_usage"]/1000

col1, col2 = st.beta_columns(2)

with col1:
    fig = px.bar(df_sel, x="batch_size", y=["throughput"],
                 labels={
                     "value": "Sequences per second",
                 },
                 title="Training Throughput")
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(df_sel, x="batch_size", y=compute_segments,
                 labels={
                     "value": "Time (seconds)",
                 },
                 title="Compute Timings")
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

if show_infer:
    col1, col2 = st.beta_columns(2)

    df_sel["infer_throughput"] = df_sel["batch_size"]/df_sel["forward"]

    with col1:
        fig = px.bar(df_sel, x="batch_size", y=["infer_throughput"],
                    labels={
                        "value": "Sequences per second",
                    },
                    title="Inference Throughput (estimated)")
        fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(df_sel, x="batch_size", y=["forward"],
                    labels={
                        "value": "Time (seconds)",
                    },
                    title="Inference Latency (estimated)")
        fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)


col1, col2 = st.beta_columns(2)

with col1:
    fig = px.bar(df_sel, x="batch_size", y=["sm_util", "tc_util", "vram_io"],
                 range_y=[0,100],
                 labels={
                     "value": "Percent (%)",
                 },
                 title="GPU Utilization", barmode='group')
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(df_sel, x="batch_size", y=["vram_usage"],
                 labels={
                     "value": "VRAM Usage (GB)",
                 },
                 title="VRAM Usage", barmode='group')
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

with st.beta_expander(label="More information", expanded=True):
    st.markdown("""**Important disclaimer**\n
Benchmarks are done by myself independently on with NVIDIA NGC PyTorch 20.10 container. The information is provided on a best-effort basis and I make no guarantees of the accuracy of the information. Feel free to [open a GitHub issue](https://github.com/tlkh/transformers-benchmarking/issues/new) if you have any questions or suggestions.""")
    st.markdown("This dashboard shows the measured performance of GPUs when training various configurations of Transformer networks, showing throughput (seq/s) and GPU memory (VRAM) usage. The idea is to allow users have an easy reference for choosing model configuration (model size/batch size/sequence length) and GPU model.")
    st.markdown("Each HuggingFace Transformer model is loaded in PyTorch as a `AutoModelForSequenceClassification` and used in a PyTorch Lightning `LightningModule` for training.")
    st.markdown("Forward and backward pass computation timings and provided by the PyTorch Lightning built-in [`SimpleProfiler`](https://pytorch-lightning.readthedocs.io/en/latest/profiler.html#enable-simple-profiling).")
    st.markdown("GPU utilization shown is likely lower than actual since measuring GPU utilization (especially using [DLProf](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/) to measure Tensor Core utilization) likely reduces GPU utilization due to overheads. This is not present in the throughput measurements.")
    st.markdown("""Systems used:
* V100 - local OEM system
* A100 - AWS p4d.24xlarge instance""")
    st.markdown("Only single GPU configuration has been tested.")
    st.markdown("All current results are using PyTorch [native automatic mixed precision](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/) and [Apex fused AdamW optimizer](https://nvidia.github.io/apex/optimizers.html).")
    st.markdown("GitHub repository: https://github.com/tlkh/transformers-benchmarking")

with st.beta_expander(label="Raw Data Table", expanded=False):
    st.dataframe(df)
