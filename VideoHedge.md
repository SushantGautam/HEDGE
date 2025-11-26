# **VideoHEDGE – Reproduction Guide (with SoccerChat Benchmark)**

This documentation explains how to reproduce the VideoHEDGE experiments with SoccerChat, including:

- Dataset extraction & caching distorted variants  
- Temperature-based answer generation using VLLM  
- Domain-specific hallucination adjudication (`Qwen3-30B-A3B`)  
- Embedding- and NLI-based clustering  
- Reliability metric computation (SE, RadFlag, VASE)  
- Variations over frame rate & pixel budgets  
- Segment-wise GPU profiling  

## 1. Environment

```bash
conda create -n hedge_video python=3.10
conda activate hedge_video

pip install vllm accelerate datasets transformers sentence-transformers
pip install pandas pyarrow tqdm
pip install hedge-bench
```

## 2. High-level pipeline

1. [`1_generate_answer.py`](https://github.com/SushantGautam/HEDGE/blob/main/VideoHedge/1_generate_answer.py) – distort & cache SoccerChat, generate answers with VLMs  
2. [`2_add_hallucination_label.py`](https://github.com/SushantGautam/HEDGE/blob/main/VideoHedge/2_add_hallucination_label.py) – add hallucination labels via Qwen3-30B-A3B  
3. [`3_cluster.py`](https://github.com/SushantGautam/HEDGE/blob/main/VideoHedge/3_cluster.py) – embedding + NLI clustering, compute ROC–AUC for SE, RadFlag, VASE  
4. [`4_distortion_varying.py`](https://github.com/SushantGautam/HEDGE/blob/main/VideoHedge/4_distortion_varying.py) – sweep distortion budgets + GPU profiling  
5. [`5_frame_pixel_vary.py`](https://github.com/SushantGautam/HEDGE/blob/main/VideoHedge/5_frame_pixel_vary.py) – vary frame rate and max pixels, regenerate answers  
6. [`4.1_add_hallucination_label.py`](https://github.com/SushantGautam/HEDGE/blob/main/VideoHedge/4.1_add_hallucination_label.py) – hallucination labels for pixel/frame variants  
7. [`5.1_cluster.py`](https://github.com/SushantGautam/HEDGE/blob/main/VideoHedge/5.1_cluster.py) – clustering and AUCs for pixel/frame variants  


Run each script with `CUDA_VISIBLE_DEVICES=... python <script>.py` according to your setup.
