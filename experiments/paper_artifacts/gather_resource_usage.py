
from transformers import pipeline
import pandas as pd
from hedge_bench.utils import apply_nli_clustering, optimize_and_apply_embed_clustering, clamp_distortions, compute_roc_aucs
import time
import json
import torch


def run_with_gpu_profile(name, func, *args, **kwargs):
    """
    Run `func(*args, **kwargs)` and measure:
      - wall-clock time
      - delta & peak GPU memory (if CUDA is available)
    Returns (result, stats_dict).
    """
    stats = {"name": name}

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    else:
        start_mem = None

    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    if cuda_available:
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    stats["time_sec"] = end_time - start_time

    if cuda_available:
        peak_mem = torch.cuda.max_memory_allocated()
        stats["gpu_peak_bytes"] = int(peak_mem)
        stats["gpu_peak_mb"] = peak_mem / (1024 ** 2)
        stats["gpu_delta_bytes"] = int(peak_mem - start_mem)
        stats["gpu_delta_mb"] = (peak_mem - start_mem) / (1024 ** 2)
    else:
        stats["gpu_peak_bytes"] = None
        stats["gpu_peak_mb"] = None
        stats["gpu_delta_bytes"] = None
        stats["gpu_delta_mb"] = None

    print(f"🔧 {name}: {stats['time_sec']:.2f} s", end="")
    if cuda_available:
        print(f" | ΔGPU ≈ {stats['gpu_delta_mb']:.1f} MB, peak ≈ {stats['gpu_peak_mb']:.1f} MB")
    else:
        print(" | CUDA not available, CPU-only timing.")
    return result, stats

    
if __name__ == '__main__':  # guard for multiprocessing
    # put NLI on GPU if available; adjust device index if needed
    device = 0 if torch.cuda.is_available() else -1
    nli = pipeline(
        "text-classification",
        model="microsoft/deberta-v2-xlarge-mnli",
        top_k=None,
        truncation=True,
        max_length=512,
        device=device,
    )

    all_roc_aucs = {}
    thresholds = {}
    profile_log = []   # list of per-call stats

    parquet = "/home/sushant/D1/HEDGE/Qwen2.5-VL-7B-Instructvqa_rad_test_answers_hallucination_label_added.parquet"
    answers_all = pd.read_parquet(parquet)

    for max_distortions in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]:
        print(f"\n🔍 Processing {parquet}, with max_distortions={max_distortions} ...")
        answers = clamp_distortions(answers_all, max_distortions=max_distortions)

        # 1) NLI clustering
        answers_nli, stats_nli = run_with_gpu_profile(
            name=f"apply_nli_clustering (max_distortions={max_distortions})",
            func=apply_nli_clustering,
            dataframe=answers,
            nli_model=nli,
            batch_size=128,
            append_question=False,
        )
        stats_nli["stage"] = "nli_clustering"
        stats_nli["max_distortions"] = max_distortions
        profile_log.append(stats_nli)

        # 2) Embed clustering – 20 trials (heavy)
        #     want answers_deep = answers.copy(deep=True) here.)
        _, stats_embed_20 = run_with_gpu_profile(
            name=f"optimize_and_apply_embed_clustering (n_trials=20, max_distortions={max_distortions})",
            func=optimize_and_apply_embed_clustering,
            df=answers,
            append_question=False,
            n_trials=20,
        )
        stats_embed_20["stage"] = "embed_clustering_ntrials_20"
        stats_embed_20["max_distortions"] = max_distortions
        profile_log.append(stats_embed_20)

        # 3) Embed clustering – 1 trial (light)
        _, stats_embed_1 = run_with_gpu_profile(
            name=f"optimize_and_apply_embed_clustering (n_trials=1, max_distortions={max_distortions})",
            func=optimize_and_apply_embed_clustering,
            df=answers,
            append_question=False,
            n_trials=1,
        )
        stats_embed_1["stage"] = "embed_clustering_ntrials_1"
        stats_embed_1["max_distortions"] = max_distortions
        profile_log.append(stats_embed_1)

    # Optional: dump to JSON for later analysis
    with open("hedge_gpu_profile.json", "w") as f:
        json.dump(profile_log, f, indent=2)
    print("\n✅ Profiling finished. Saved to hedge_gpu_profile.json")