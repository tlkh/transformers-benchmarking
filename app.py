import streamlit as st
st.set_page_config(
    page_title="Transformers Benchmarks",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded",
)
import pandas as pd
import plotly.express as px

SEQ_LEN_LIST = [128, 512]
MODEL_LIST = ["distilroberta-base"]
GPU_LIST = ["v100-16gb-300w"]
COMPUTE_LIST = ["amp"]

st.sidebar.markdown("# 🤗 Transformers Benchmarking")
st.sidebar.markdown("**Options**")
model_name = st.sidebar.selectbox("Model Name", MODEL_LIST)
gpu_name = st.sidebar.selectbox("GPU", GPU_LIST)
compute_name = st.sidebar.selectbox("Compute Type", COMPUTE_LIST)
seq_len = st.sidebar.selectbox("Sequence Length", SEQ_LEN_LIST)

df = pd.read_csv("./results.csv")

def sum_and_reciprocal_to_col(frame, new_col_name, list_of_cols_to_sum):
    frame[new_col_name] = frame[list_of_cols_to_sum].astype(float).sum(axis=1)
    frame[new_col_name] = frame["batch_size"] * (1/frame[new_col_name])
    return(frame)

df = sum_and_reciprocal_to_col(df, "throughput", ["cpu_time", "forward", "backward"])
df["vram_usage"] = df["vram_usage"]/1000
df.sort_values(by="batch_size", inplace=True)
df_sel = df[df["seq_len"] == seq_len]
df_sel = df_sel[df["gpu"] == gpu_name]
df_sel = df_sel[df["model"] == model_name]
df_sel = df_sel[df["compute"] == compute_name]

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
    fig = px.bar(df_sel, x="batch_size", y=["cpu_time", "forward", "backward"],
                 labels={
                     "value": "Time (seconds)",
                 },
                 title="Compute Timings")
    fig.update_xaxes(type='category')
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.beta_columns(2)

with col1:
    fig = px.bar(df_sel, x="batch_size", y=["sm_util", "tc_util", "vram_io"],
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

with st.beta_expander(label="Data Table", expanded=False):
    st.dataframe(df)
