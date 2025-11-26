import pandas as pd
from pathlib import Path
from hedge_bench.utils import add_hallucination_labels_vllm

paths = [
    # "/home/sushant/D1/HEDGE_video/pixelvar_SoccerChat_500_SoccerChat-Qwen2VL-FT_max_pixels10000_ALL.parquet",
    # "/home/sushant/D1/HEDGE_video/pixelvar_SoccerChat_500_SoccerChat-Qwen2VL-FT_max_pixels40000_ALL.parquet",
    # "/home/sushant/D1/HEDGE_video/pixelvar_SoccerChat_500_SoccerChat-Qwen2VL-FT_max_pixels160000_ALL.parquet",
    "/home/sushant/D1/HEDGE_video/pixelvar_SoccerChat_500_SoccerChat-Qwen2VL-FT_max_pixels250000_ALL.parquet",
]

MODEL = "Qwen/Qwen3-30B-A3B"

if __name__ == "__main__":
    for p in paths:
        p = Path(p)
        print(f"\n=== Processing: {p} ===")

        df = pd.read_parquet(p)
        m = df["variant_name"] == "EventClassification"
        df.loc[m, ["true_answer", "question"]] = pd.DataFrame(
            {
                "true_answer": df.loc[m, "description"],
                "question": "Identify the key event shown in the clip.",
            },
            index=df.index[m],
        )
        df = df.drop_duplicates(subset=["media", "question", "true_answer"])

        df_h = add_hallucination_labels_vllm(
            df,
            model_name=MODEL,
            dtype="auto",
            add_description=False,
            domain="sports",
        )
        out_path = p.with_name(p.stem + "_with_hall_labels.parquet")
        df_h.to_parquet(out_path, index=False)
        print(f"Saved → {out_path}")
