if __name__ == '__main__':
    import random
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from datasets import load_dataset
    from transformers import pipeline

    # --- utils / algorithms (only what we actually use) ---
    from utils import (
        generate_and_cache_dataset,
        generate_answers, generate_answers_vllm, to_openai_multimodal_payload,
        apply_nli_clustering,
        get_embeddings_batch,
        apply_embed_clustering,
        build_message_for_evaluation,
        evaluator_struct_output_schema,
        run_vllm_batch_from_list,
        parse_vllm_outputs_from_evaluator,
        compute_vqa_rad_aucs,
        optimized_cluster_threshold,
        PROMPT_VARIANTS
    )

    random.seed(42)
    np.random.seed(42)

    min_temp, max_temp,  dataset_id = 0.1, 1.0, "vqa_rad_test"
    n_samples = 10
    generated_data = generate_and_cache_dataset(dataset_id=dataset_id, num_samples=n_samples, vqa_dict=None, force_regenerate=False, n_jobs=40)[:500]
    # breakpoint()
    # df = generate_answers(
    #     generated_data,
    #     n_answers_high=n_samples,
    #     min_temp=min_temp,
    #     max_temp=max_temp,
    #     prompt_variants=PROMPT_VARIANTS,
    # )

    df = generate_answers_vllm(
        generated_data,
        n_answers_high=n_samples,
        min_temp=min_temp,
        max_temp=max_temp,
        prompt_variants=PROMPT_VARIANTS,
        model="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    gt_map = {i: d["answer"] for i, d in enumerate(tqdm(load_dataset("flaviagiammarino/vqa-rad", split="test")))}
    df["answer"] = df["idx_img"].map(gt_map)

    ## add hallucination label using vllm
    inputs = [
        {
            "custom_id": f"hedge-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": build_message_for_evaluation(row, add_kvasir_description=False),
                "max_completion_tokens": 1000,
            },
        }
        for i, row in df.iterrows()
    ]
    outputs = run_vllm_batch_from_list(
        inputs=inputs,
        model="Qwen/Qwen3-30B-A3B",
        allowed_media=None,
        extra_cli_args={
            "reasoning-parser": "qwen3",
            "structured-outputs-config": evaluator_struct_output_schema,
            "dtype": "bfloat16",
            "tensor-parallel-size": 1,
            "gpu-memory-utilization": 0.90,
            "enable-prefix-caching": True,
            "max-model-len": 2500,
            "mm-processor-kwargs": '{"max_pixels": 1003520}',  # 1120*896
            "override-generation-config": '{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0}',
        },
    )
    hall_map = parse_vllm_outputs_from_evaluator(outputs)
    df["hallucination_label"] = df.apply(lambda row: hall_map[f"hedge-{row.name}"], axis=1)
    ## end add hallucination label using vllm


    # 6) Embedding clustering (optimize threshold, then compute AUCs)
    unique_answers = list({
        a
        for r in df.apply(
            lambda r: [d["ans"] for d in r["original_high_temp"]]
            + [d["ans"] for d in r["distorted_high_temp"]]
            + [r["original_low_temp"]["ans"]],
            axis=1,
        )
        for a in r
    })
    embeddings = get_embeddings_batch(unique_answers, batch_size=16)
    cache = dict(zip(unique_answers, embeddings))
    embedding_cached_fn = lambda x, _cache=cache: _cache.get(x)

    threshold, _ = optimized_cluster_threshold(
        df,
        embedding_cached_fn,
        metric_path=("default", "metrics_embed", "SE"),
        threshold_range=(0.8, 0.99),
        n_trials=20,
        debug=False,
    )
    df_embed = apply_embed_clustering(df, embedding_cached_fn, threshold=threshold)

    # 5) AUCs after NLI clustering
    nli = pipeline(
        "text-classification",
        model="microsoft/deberta-large-mnli",
        top_k=None,
        truncation=True,
    )
    df_nli_embed = apply_nli_clustering(df_embed, nli, batch_size=512)

    aucs = compute_vqa_rad_aucs(df_nli_embed)
    print("\nAUCs (Embedding clustering's optimal threshold=%.3f):" % threshold)
    print(aucs)