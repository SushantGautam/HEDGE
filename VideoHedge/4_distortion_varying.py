from transformers import pipeline
import pandas as pd
from hedge_bench.utils import apply_nli_clustering, optimize_and_apply_embed_clustering, clamp_distortions, compute_roc_aucs
import time
import torch


def run_with_gpu_profile(name, func, *args, **kwargs):
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
        stats["gpu_peak_mb"] = peak_mem / (1024**2)
        stats["gpu_delta_bytes"] = int(peak_mem - start_mem)
        stats["gpu_delta_mb"] = (peak_mem - start_mem) / (1024**2)
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


if __name__ == "__main__":
    device = 0 if torch.cuda.is_available() else -1
    nli = pipeline(
        "text-classification",
        model="microsoft/deberta-v2-xlarge-mnli",
        top_k=None,
        truncation=True,
        max_length=512,
    )

    all_roc_aucs = {}
    thresholds = {}
    profile_log = []

    parquet = "/home/sushant/D1/HEDGE_video/gen_SoccerChat_500_SoccerChat-Qwen2VL-FT_0.1_1.0_10_ALL_with_hall_labels.parquet"
    answers_all = pd.read_parquet(parquet)

    for max_distortions in [6]:
        print(f"\n🔍 Processing {parquet}, max_distortions={max_distortions} ...")
        answers = clamp_distortions(answers_all, max_distortions=max_distortions)

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

        res_, stats_embed_20 = run_with_gpu_profile(
            name=f"optimize_and_apply_embed_clustering (n_trials=20, max_distortions={max_distortions})",
            func=optimize_and_apply_embed_clustering,
            df=answers_nli,
            append_question=False,
            metric_path=("VideoQA", "metrics_embed", "SE"),
            n_trials=20,
        )
        answers_embed, threshold, _ = res_
        stats_embed_20["stage"] = "embed_clustering_ntrials_20"
        stats_embed_20["max_distortions"] = max_distortions
        profile_log.append(stats_embed_20)

        _, stats_embed_1 = run_with_gpu_profile(
            name=f"optimize_and_apply_embed_clustering (n_trials=1, max_distortions={max_distortions})",
            func=optimize_and_apply_embed_clustering,
            df=answers_nli,
            append_question=False,
            metric_path=("VideoQA", "metrics_embed", "SE"),
            n_trials=1,
        )
        stats_embed_1["stage"] = "embed_clustering_ntrials_1"
        stats_embed_1["max_distortions"] = max_distortions
        profile_log.append(stats_embed_1)

        aucs = compute_roc_aucs(answers_embed)
        all_roc_aucs[max_distortions] = aucs
        thresholds[max_distortions] = threshold
        print(f"💡 max_distortions={max_distortions} AUCs: {aucs}")
