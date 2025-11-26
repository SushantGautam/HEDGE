import pandas as pd

if __name__ == "__main__":
    from transformers import pipeline
    from hedge_bench.utils import (
        optimize_and_apply_embed_clustering,
        compute_roc_aucs,
    )

    nli = pipeline(
        "text-classification",
        model="microsoft/deberta-large-mnli",
        top_k=None,
        truncation=True,
        max_length=512,
    )

    files = [
        # "/home/sushant/D1/HEDGE_video/pixelvar_SoccerChat_500_SoccerChat-Qwen2VL-FT_max_pixels10000_ALL_with_hall_labels.parquet",
        # "/home/sushant/D1/HEDGE_video/pixelvar_SoccerChat_500_SoccerChat-Qwen2VL-FT_max_pixels40000_ALL_with_hall_labels.parquet",
        # "/home/sushant/D1/HEDGE_video/gen_SoccerChat_500_SoccerChat-Qwen2VL-FT_0.1_1.0_10_ALL_with_hall_labels.parquet",
        # "/home/sushant/D1/HEDGE_video/pixelvar_SoccerChat_500_SoccerChat-Qwen2VL-FT_max_pixels160000_ALL_with_hall_labels.parquet",
        # "/home/sushant/D1/HEDGE_video/pixelvar_SoccerChat_500_SoccerChat-Qwen2VL-FT_max_pixels250000_ALL_with_hall_labels.parquet",
    ]

    aucs_list = []
    for file in files:
        print(f"\n=== Processing: {file} ===")
        answers = pd.read_parquet(file)
        print(answers.groupby(["variant_name", "hallucination_label"]).size().to_dict())

        answers["hallucination_label"] = answers["hallucination_label"].apply(
            lambda x: 0 if x == -1 else x
        )

        print("🧩 Embedding-based clustering...")
        answers_embed, threshold, _ = optimize_and_apply_embed_clustering(
            answers,
            append_question=False,
            metric_path=("VideoQA", "metrics_embed", "SE"),
        )
        print(f"Embedding clustering threshold = {threshold:.3f}")

        aucs = compute_roc_aucs(answers_embed)
        aucs["score_distribution"] = (
            answers["hallucination_label"].value_counts().to_dict()
        )
        print("AUCs:", aucs)
        aucs_list.append((file, aucs))

    print("\nAll AUCs:")
    print(aucs_list)
