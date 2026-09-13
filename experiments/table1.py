import pandas as pd
from transformers import pipeline

from hedge_bench.utils import (
    apply_nli_clustering,
    compute_roc_aucs,
    optimize_and_apply_embed_clustering, clamp_distortions
)


answers = pd.read_parquet("../results/archive/Qwen2.5-VL-7B-Instructvqa_rad_test_answers_hallucination_label_added.parquet")
answers = clamp_distortions(answers, max_distortions=10)

# 1) 1 also try clustering with an NLI model
print("🧮 Applying NLI-based clustering with DeBERTa...")
nli = pipeline("text-classification", model="microsoft/deberta-v2-xlarge-mnli", top_k=None, truncation=True, max_length=512)


# answers = pd.DataFrame([{
#     "original_high_temp": [
#         {"ans": a, "logprob": [0]}
#         for a in [
#             "vagina","Esophagus","esophagus","esophagus","stomach",
#             "gastric","esophagus","esophagus","stomach","Esophagus"
#         ]
#     ],
#     "distorted_high_temp": [
#         {"ans": a, "logprob": [0]}
#         for a in [
#             "esophagus","esophagus","esophagus","CVP","STOMACH",
#             "Esophagus","esophagus","incision site","stomach","Intestine"
#         ]
#     ],
#     "original_low_temp": {"ans": "esophagus", "logprob": [0]},
# }])


# print(apply_nli_clustering(answers, nli, batch_size=512,  append_question=False).iloc[0].cluster_nli)

# breakpoint()

answers_nli = apply_nli_clustering(answers, nli, batch_size=512,  append_question=False)

# 2) Cluster by embeddings
print("🧩 Performing embedding-based clustering...")
answers_nli_embed, threshold, _ = optimize_and_apply_embed_clustering(answers_nli,  append_question=False)
print(f"📈 Embedding clustering complete! Optimal threshold = {threshold:.3f} 🎯")

aucs= compute_roc_aucs(answers_nli_embed)
print(f"💡 Embedding clustering (threshold={threshold:.3f}) AUCs:")
print(aucs)

print("🎉 Pipeline completed successfully! ✅✨")
