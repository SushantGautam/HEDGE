import pandas as pd
from transformers import pipeline
from hedge_bench.utils import (
    apply_nli_clustering,
    optimize_and_apply_embed_clustering,
    compute_roc_aucs,
)

if __name__ == "__main__":
    nli = pipeline(
        "text-classification",
        model="microsoft/deberta-large-mnli",
        top_k=None,
        truncation=True,
        max_length=512,
    )

    files = [
        "/home/sushant/D1/HEDGE_video/gen_SoccerChat_500_Qwen2-VL-7B-Instruct_0.1_1.0_10_ALL_with_hall_labels.parquet",
        "/home/sushant/D1/HEDGE_video/gen_SoccerChat_500_Qwen2.5-VL-7B-Instruct_0.1_1.0_10_ALL_with_hall_labels.parquet",
        "/home/sushant/D1/HEDGE_video/gen_SoccerChat_500_SoccerChat-Qwen2VL-FT_0.1_1.0_10_ALL_with_hall_labels.parquet",
    ]

    aucs_list = []
    for file in files:
        print(f"\n=== Processing: {file} ===")
        answers = pd.read_parquet(file)

        print("🧩 Embedding-based clustering...")
        answers_embed, threshold, _ = optimize_and_apply_embed_clustering(
            answers,
            append_question=False,
            metric_path=("VideoQA", "metrics_embed", "SE"),
        )
        print(f"Embedding clustering threshold = {threshold:.3f}")

        print("🧮 NLI-based clustering...")
        answers_embed_clustered = apply_nli_clustering(
            answers_embed,
            nli,
            batch_size=128,
            append_question=False,
        )

        aucs = compute_roc_aucs(answers_embed_clustered)
        print("AUCs:", aucs)
        aucs_list.append((file, aucs))

    print("\nAll AUCs:")
    print(aucs_list)
