# /home/sushant/D1/HEDGE/redo_experimnts_2026/llava-med-v1.5-mistral-7b-hfkvasir_vqa_x1_test_answers_hallucination_label_added.parquet.json
import pandas as pd
from sklearn.metrics import roc_auc_score



def compute_roc_aucs_by_variant(df):
    results = []

    for variant, group in df.groupby("variant_name"):
        y_true = group["y_true"].values # 1 = correct, 0 = hallucination
        res = {"variant_name": variant}

        # MaxProb (higher = better)
        if "maxprob" in group.columns:
            res["roc_auc_maxprob"] = roc_auc_score(y_true, group["maxprob"].values)

        # Entropy (higher = worse → invert)
        if "predictive_entropy" in group.columns:
            res["roc_auc_predictive_entropy"] = roc_auc_score(
                y_true, -group["predictive_entropy"].values
            )

        results.append(res)

    return pd.DataFrame(results)

core_experiment_outputs = [
        {"model": "llava-med-v1.5-mistral-7b", "dataset": "vqa_rad",
        "file": "../results/llava-med-v1.5-mistral-7b-hfvqa_rad_test_answers_hallucination_label_added.parquet"},

        {"model": "llava-med-v1.5-mistral-7b", "dataset": "kvasir_vqa_x1",
        "file": "../results/llava-med-v1.5-mistral-7b-hfkvasir_vqa_x1_test_answers_hallucination_label_added.parquet"},

        {"model": "medgemma-4b-it", "dataset": "vqa_rad",
        "file": "../results/medgemma-4b-itvqa_rad_test_answers_hallucination_label_added.parquet"},

        {"model": "medgemma-4b-it", "dataset": "kvasir_vqa_x1",
        "file": "../results/medgemma-4b-itkvasir_vqa_x1_test_answers_hallucination_label_added.parquet"},

        {"model": "Qwen2.5-VL-7B-Instruct", "dataset": "vqa_rad",
        "file": "../results/Qwen2.5-VL-7B-Instructvqa_rad_test_answers_hallucination_label_added.parquet"},

        {"model": "Qwen2.5-VL-7B-Instruct", "dataset": "kvasir_vqa_x1",
        "file": "../results/Qwen2.5-VL-7B-Instructvqa_kvasir_vqa_x1_test_answers_hallucination_label_added.parquet"},

        {"model": "clustered-v1", "dataset": "POPE",
        "file": "../results/caches/tmp_2026_pope.py.answers.clustered-v1.pickle"},

]

import numpy as np
import pandas as pd

def compute_metrics(logprobs):
    logprobs = np.array(logprobs)

    # Convert to probabilities
    probs = np.exp(logprobs)

    # Normalize (important if logprobs are not perfectly normalized)
    probs = probs / probs.sum()

    # Max probability
    maxprob = probs.max()

    # Predictive entropy
    entropy = -(probs * np.log(probs + 1e-12)).sum()

    return maxprob, entropy


def extract_metrics(x):
    logprobs = x["logprob"]
    return pd.Series(compute_metrics(logprobs), index=["maxprob", "entropy"])


import json
answer_only = {}
answer_question_passed = {}
for output in core_experiment_outputs:
    for append_question_label in [False ]:
        dataset_id = output['dataset']
        filename = output['file']

        if "_pope" in filename:
            answers = pd.read_pickle(filename)
        else:
            answers = pd.read_parquet(filename)
        if "_pope" in filename and append_question_label == True:
            continue
    
        if append_question_label==False:
            finalfilename = f"{filename}"
        else:
            finalfilename = f"{filename}_append_question.parquet"
        answers[["maxprob", "predictive_entropy"]] = answers["original_low_temp"].apply(extract_metrics)
        print(f"Model: {output['model']}, Dataset: {output['dataset']}")
        metrics = compute_roc_aucs_by_variant(answers)
        print(metrics)
        # breakpoint()
        
# breakpoint()