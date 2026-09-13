
from transformers import pipeline
import pandas as pd
from hedge_bench.utils import apply_nli_clustering, optimize_and_apply_embed_clustering, clamp_distortions, compute_roc_aucs

if __name__ == '__main__': # guard  for multiprocessing
    nli = pipeline("text-classification", model="microsoft/deberta-v2-xlarge-mnli", top_k=None, truncation=True, max_length=512)
    all_roc_aucs = {}
    thresholds = {}
    parquet= "results/archive/Qwen2.5-VL-7B-Instructvqa_rad_test_answers_hallucination_label_added.parquet"
    answers_all = pd.read_parquet(parquet)
    for max_distortions in [30]:
        answers = clamp_distortions(answers_all, max_distortions=max_distortions)
        print(f"🔍 Processing {parquet}, with max_distortions={max_distortions} ...")
        # answers_nli= apply_nli_clustering(answers, nli, batch_size=128, append_question=False)
        answers_nli_embed, thr, _ = optimize_and_apply_embed_clustering(answers, append_question=False, model_name="NeuML/pubmedbert-base-embeddings")
        print(f"🧩 Optimal embedding clustering threshold: {thr:.3f} 🎯")   
        roc_aucs = compute_roc_aucs(answers_nli_embed)
        print(f"📊 {parquet} | max_distortions={max_distortions} |  {roc_aucs}")
        key = f"qwen_vqa_rad_max_distortions_{max_distortions}"
        all_roc_aucs[key] = roc_aucs
        thresholds[key] = thr
    print("💡 All ROC AUCs:")
    for k, v in all_roc_aucs.items():
        print(f"{k}: {v}")
    print("💡 All thresholds:")
    for k, v in thresholds.items():
        print(f"{k}: {v}")
