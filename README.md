# HEDGE: Hallucination Estimation via Dense Geometric Entropy

HEDGE provides the code and Python package that accompany the paper **"HEDGE: Hallucination Estimation via Dense Geometric Entropy for Medical VQA with Vision-Language Models."** The library offers utilities for sampling answers from multimodal models, clustering them through logical and embedding-based strategies, and computing hallucination detection metrics across benchmarks such as VQA-RAD and KvasirVQA.

## Installation

The utilities published in this repository are available on PyPI as [`hedge-bench`](https://pypi.org/project/hedge-bench/).

```bash
pip install hedge-bench
```

You can also install the package from source by `pip install git+https://github.com/SushantGautam/HEDGE.git`.

## Quickstart

The snippet below shows a minimal end-to-end example adapted from `tmp_test.py`, demonstrating how to generate answer samples, apply both embedding- and NLI-based clustering, and evaluate hallucination detection metrics.

```python
from datasets import load_dataset
from transformers import pipeline

from hedge_bench.utils import (
    PROMPT_VARIANTS,
    add_hallucination_labels_vllm,
    apply_nli_clustering,
    compute_roc_aucs,
    generate_distortion_and_cache_dataset,
    generate_answers,
    optimize_and_apply_embed_clustering,
)

# 1) Prepare a small VQA-RAD subset
n_samples = 3
vqa_dict = [
    {"idx": i, "image": sample["image"], "question": sample["question"], "answer": sample["answer"]}
    for i, sample in enumerate(load_dataset("flaviagiammarino/vqa-rad", split="test"))
][:10]

generated = generate_distortion_and_cache_dataset(
    dataset_id="vqa_rad_test",
    num_samples=n_samples,
    vqa_dict=vqa_dict,
    force_regenerate=False,
    n_jobs=40,
)

# 2) Sample answers from a vision-language model
answers = generate_answers(
    generated,
    n_answers_high=n_samples,
    min_temp=0.1,
    max_temp=1.0,
    prompt_variants=PROMPT_VARIANTS,
    model="Qwen/Qwen2.5-VL-7B-Instruct",
)

# 3) Label hallucinations using a VLM judge and cluster by embeddings
answers = add_hallucination_labels_vllm(answers)
answers_embed, threshold, _ = optimize_and_apply_embed_clustering(answers)

# 4) Optionally, also try clustering with an NLI model and compute ROC AUCs
nli = pipeline("text-classification", model="microsoft/deberta-large-mnli", top_k=None, truncation=True)
answers_clustered = apply_nli_clustering(answers_embed, nli, batch_size=768)

aucs = compute_roc_aucs(answers_clustered)
print(f"Embedding clustering optimal threshold = {threshold:.3f}")
print(aucs)
```

## Project layout

- `hedge_bench/algorithms.py` – reference implementations of uncertainty estimators, clustering strategies, and scoring utilities.
- `hedge_bench/utils.py` – high-level helper functions for dataset caching, answer generation, labeling, and evaluation (as used in the quickstart example).

## Citation

If you use HEDGE in your work, please cite the associated paper.

```
HEDGE: Hallucination Estimation via Dense Geometric Entropy for Medical VQA with Vision-Language Models
```

## License

This project is released under the MIT License. See `LICENSE` for details.
