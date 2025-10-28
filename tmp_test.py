if __name__ == '__main__':
    import random
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from datasets import load_dataset
    from transformers import pipeline

    from utils import (
        generate_and_cache_dataset,
        generate_answers,
        add_hallucination_labels_vllm,
        optimize_and_apply_embed_clustering,
        apply_nli_clustering,
        compute_roc_aucs,
        PROMPT_VARIANTS,
    )

    min_temp, max_temp,  dataset_id = 0.1, 1.0, "vqa_rad_test"
    n_samples = 3

    vqa_dict = [ {"idx": i, "image": d["image"], "question": d["question"], "answer": d["answer"]} for i, d in enumerate(tqdm(load_dataset("flaviagiammarino/vqa-rad", split="test")))]
    generated_data = generate_and_cache_dataset(dataset_id=dataset_id, num_samples=n_samples, vqa_dict=vqa_dict, force_regenerate=False, n_jobs=40)[:10]  # just select 10 samples for testing
    breakpoint()
    df = generate_answers(
        generated_data,
        n_answers_high=n_samples,
        min_temp=min_temp,
        max_temp=max_temp,
        prompt_variants=PROMPT_VARIANTS,
        model="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    breakpoint()
    gt_map = {i: d["answer"] for i, d in enumerate(tqdm(load_dataset("flaviagiammarino/vqa-rad", split="test")))}
    df["answer"] = df["idx_img"].map(gt_map)

    df = add_hallucination_labels_vllm(df)

    df_embed, threshold, _ = optimize_and_apply_embed_clustering(df)

    # 5) AUCs after NLI clustering
    nli = pipeline(
        "text-classification",
        model="microsoft/deberta-large-mnli",
        top_k=None,
        truncation=True,
    )
    df_nli_embed = apply_nli_clustering(df_embed, nli, batch_size=768)

    aucs = compute_roc_aucs(df_nli_embed)
    print("\nAUCs (Embedding clustering's optimal threshold=%.3f):" % threshold)
    print(aucs)