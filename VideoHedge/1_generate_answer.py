from datasets import load_dataset
from huggingface_hub import snapshot_download
from hedge_bench.utils import distort_and_cache_dataset, generate_answers
import random
from pathlib import Path
import pandas as pd

random.seed(42)

ds = load_dataset("SimulaMet/SoccerChat", split="train").remove_columns("video")
keep = set(ds.filter(lambda x: len(x["events"]) == 1).shuffle(seed=42)["path"][:500])
ds_filtered = ds.filter(lambda x: x["path"] in keep)

snapshot_download(
    repo_id="SimulaMet/SoccerChat",
    repo_type="dataset",
    allow_patterns=[f"videos/{path}" for path in keep],
    local_dir="/home/sushant/D1/HEDGE_video/SoccerChat_assets",
)

soccerchat_dict = [
    {
        "idx": i,
        "video": "/home/sushant/D1/HEDGE_video/SoccerChat_assets/videos/" + d["path"],
        "question": d["query"],
        "answer": d["response"],
        "description": d["events"][0],
    }
    for i, d in enumerate(ds_filtered)
]

PROMPT_VARIANTS_base = {
    "EventClassification": [
        {
            "role": "system",
            "content": "You are a sports video reasoning assistant. Given a short video clip and a user question, provide an answer that is concise and directly addresses exactly what is asked. Ground the answer strictly in the video content. When referring to teams or players, always use jersey colors. Do not give explanations or extra text.",
        },
        {"role": "user", "content": "<video> Identify the key event shown in the clip."},
    ],
    "VideoQA": [
        {
            "role": "system",
            "content": "You are a sports video reasoning assistant. Given a short video clip and a user question, provide an answer that is concise and directly addresses exactly what is asked. Ground the answer strictly in the video content. When referring to teams or players, always use jersey colors. Do not give explanations or extra text.",
        },
        {"role": "user", "content": "<video> {r.question}"},
    ],
}

if __name__ == "__main__":
    min_temp, max_temp = 0.1, 1.0
    n_samples = 10
    dataset_id = "SoccerChat_500"

    generated_data = distort_and_cache_dataset(
        dataset_id=dataset_id,
        num_samples=n_samples,
        vqa_dict=soccerchat_dict,
        force_regenerate=False,
        n_jobs=30,
    )

    chunk_size = 300

    models = [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        # "/home/sushant/D1/HEDGE_video/SoccerChat-Qwen2VL-FT",
    ]

    for model in models:
        all_dfs = []
        model_save_ = model.split("/")[-1]
        base_name = f"gen_{dataset_id}_{model_save_}_{min_temp}_{max_temp}_{n_samples}"
        base_path = Path(".")
        if "soccerchat" in model_save_.lower():
            PROMPT_VARIANTS = {
                k: [m for m in v if m.get("role") != "system"]
                for k, v in PROMPT_VARIANTS_base.items()
            }
        else:
            PROMPT_VARIANTS = PROMPT_VARIANTS_base

        final_out = base_path / f"{base_name}_ALL.parquet"
        if final_out.exists():
            print(f"{final_out} exists, skipping {model}")
            continue

        for start in range(0, len(generated_data), chunk_size):
            end = min(start + chunk_size, len(generated_data))
            chunk_idx = start // chunk_size
            chunk = generated_data[start:end]
            chunk_fname = base_path / f"{base_name}_chunk_{chunk_idx}.parquet"

            if chunk_fname.exists():
                df_chunk = pd.read_parquet(chunk_fname)
            else:
                df_chunk = generate_answers(
                    chunk,
                    n_answers_high=n_samples,
                    min_temp=min_temp,
                    max_temp=max_temp,
                    prompt_variants=PROMPT_VARIANTS,
                    model=model,
                    extra_cli_args={
                        "limit-mm-per-prompt": {"image": 0, "video": 1},
                        "tensor-parallel-size": 4,
                        "max-model-len": 2000,
                        "mm-processor-kwargs": {"max_pixels": 100352},
                        "media-io-kwargs": {"video": {"num_frames": 24}},
                    },
                )
                df_chunk.to_parquet(chunk_fname, index=False)
            all_dfs.append(df_chunk)

        answer_df = pd.concat(all_dfs, ignore_index=True)
        answer_df.to_parquet(final_out, index=False)

        print("Cleaning temporary chunk files...")
        for start in range(0, len(generated_data), chunk_size):
            chunk_idx = start // chunk_size
            chunk_fname = base_path / f"{base_name}_chunk_{chunk_idx}.parquet"
            if chunk_fname.exists():
                try:
                    chunk_fname.unlink()
                except Exception as e:
                    print(f"Could not delete {chunk_fname}: {e}")
