if __name__ == '__main__': # guard  for multiprocessing
    import os
    # (hedge) sushant@g002:~/D1/HEDGE$ 
    core_experiment_outputs = [
        # {"model": "llava-med-v1.5-mistral-7b", "dataset": "vqa_rad",
        # "file": "../llava-med-v1.5-mistral-7b-hfvqa_rad_test_answers_hallucination_label_added.parquet"},

        # {"model": "llava-med-v1.5-mistral-7b", "dataset": "kvasir_vqa_x1",
        # "file": "../llava-med-v1.5-mistral-7b-hfkvasir_vqa_x1_test_answers_hallucination_label_added.parquet"},

        # {"model": "medgemma-4b-it", "dataset": "vqa_rad",
        # "file": "../medgemma-4b-itvqa_rad_test_answers_hallucination_label_added.parquet"},

        # {"model": "medgemma-4b-it", "dataset": "kvasir_vqa_x1",
        # "file": "../medgemma-4b-itkvasir_vqa_x1_test_answers_hallucination_label_added.parquet"},

        {"model": "Qwen2.5-VL-7B-Instruct", "dataset": "vqa_rad",
        "file": "../results/Qwen2.5-VL-7B-Instructvqa_rad_test_answers_hallucination_label_added.parquet"},

        {"model": "Qwen2.5-VL-7B-Instruct", "dataset": "kvasir_vqa_x1",
        "file": "../results/Qwen2.5-VL-7B-Instructvqa_kvasir_vqa_x1_test_answers_hallucination_label_added.parquet"},
    ]

    ## I suspect that add_hallucination_labels_vllm had problem with keys, so redo and see
    # can also chekc distribution of hallucination labels across varients
    # read parquet files
    import pandas as pd
    import json
    from transformers import pipeline
    from hedge_bench.utils import (
            # PROMPT_VARIANTS,
            add_hallucination_labels_vllm,
            apply_nli_clustering,
            compute_roc_aucs,
            distort_and_cache_dataset,
            generate_answers,
            optimize_and_apply_embed_clustering,
            build_message_for_evaluation_medical
    )
    for exp in core_experiment_outputs:
        for append_question_label in [False, True ]:
            dataset_id = exp['dataset']
            filename = exp['file']
            answers = pd.read_parquet(filename)
            if append_question_label==False:
                finalfilename = f"{filename}"
            else:
                finalfilename = f"{filename}_append_question.parquet"
            # answers = answers.sample(n=100, random_state=42)  # take 100 samples only for quick testing
            # answers['media']= answers['image'].astype(str)
            # # 3) Label hallucinations using a VLM judge and cluster by embeddings
            # answers = add_hallucination_labels_vllm(answers, model_name="Qwen/Qwen3-30B-A3B", dtype="auto", add_description=True if 'kvasir' in dataset_id else False, message_builder = build_message_for_evaluation_medical)
            # # uses   File "/global/D1/homes/sushant/HEDGE/hedge_bench/utils.py", line 925, in add_hallucination_labels_vllm
            # # df["_hedge_key"] = (df["media"].astype(str) + "||" +df["question"].astype(str) + "||" +df["true_answer"].astype(str))+"||"+df["variant_name"].astype(str)
            # print(answers.groupby(['variant_name', 'hallucination_label']).size().reset_index(name='count'))
            print("running for ", finalfilename)
            if os.path.exists(finalfilename):
                print(f"{finalfilename} already exists, skipping...")
                continue
            answers_embed, threshold, _ = optimize_and_apply_embed_clustering(answers,  append_question=append_question_label)

            # 4) Optionally, also try clustering with an NLI model and compute ROC AUCs
            nli = pipeline("text-classification", model="microsoft/deberta-large-mnli", top_k=None, truncation=True)
            answers_clustered = apply_nli_clustering(answers_embed, nli, batch_size=128, append_question=append_question_label)

            aucs = compute_roc_aucs(answers_clustered)
            print(f"Embedding clustering optimal threshold = {threshold:.3f}")
            # print(aucs)
            # save in redo_experimnts_2026, answers_clusteredas dataset_id.json and answers_clustered as dataset_id.parquet
            with open(f"../results/{finalfilename}.json", "w", encoding="utf-8") as f:
                json.dump(aucs, f, ensure_ascii=False, indent=2)
            answers_clustered.to_parquet(f"../results/{finalfilename}", index=False)