if __name__ == '__main__': # guard  for multiprocessing
    from datasets import load_dataset
    from transformers import pipeline

    from hedge_bench.utils import (
        PROMPT_VARIANTS,
        add_hallucination_labels_vllm,
        apply_nli_clustering,
        compute_roc_aucs,
        generate_distortion_and_cache_dataset,
        generate_answers,
        optimize_and_apply_embed_clustering,
    )

    # 1) Prepare a small VQA-RAD subset
    print("💾 Starting to cache data for VQA-RAD dataset...")
    n_samples = 3
    vqa_dict = [
        {"idx": i, "image": sample["image"], "question": sample["question"], "answer": sample["answer"]}
        for i, sample in enumerate(load_dataset("flaviagiammarino/vqa-rad", split="test"))
    ]
    generated = generate_distortion_and_cache_dataset(
        dataset_id="vqa_rad_test",  # will cache with this name
        num_samples=n_samples,
        vqa_dict=vqa_dict,
        force_regenerate=False,
        n_jobs=2
        )[:10]  # take 10 samples only
    print("✅ Dataset prepared successfully! 📊")

    # 2) Sample answers from a vision-language model
    print("🤖 Generating answers ...")
    answers = generate_answers(
        generated,
        n_answers_high=n_samples,
        min_temp=0.1,
        max_temp=1.0,
        prompt_variants=PROMPT_VARIANTS,
        model="Qwen/Qwen2-VL-2B-Instruct",
        extra_cli_args={"dtype":"auto"},
    )
    print("🧠 Answers generated successfully! 💬")

    # 3) Label hallucinations using a VLM judge
    print("🔍 Labeling hallucinations with Qwen3")
    answers = add_hallucination_labels_vllm(answers, model_name="Qwen/Qwen3-4B-Instruct-2507", dtype="auto")

    print("🏷️ Hallucination labels added! ✅")

    # 4) Cluster by embeddings
    print("🧩 Performing embedding-based clustering...")
    answers_embed, threshold, _ = optimize_and_apply_embed_clustering(answers)
    print(f"📈 Embedding clustering complete! Optimal threshold = {threshold:.3f} 🎯")

    # 5) Optionally, also try clustering with an NLI model
    print("🧮 Applying NLI-based clustering with DeBERTa...")
    nli = pipeline("text-classification", model="microsoft/deberta-large-mnli", top_k=None, truncation=True, max_length=512)
    answers_clustered = apply_nli_clustering(answers, nli, batch_size=64)
    print("📊 NLI clustering complete! 🧠")

    aucs_nli = compute_roc_aucs(answers_clustered)
    print("🧾 NLI clustering AUCs:")
    print(aucs_nli)

    aucs_embed = compute_roc_aucs(answers_embed)
    print(f"💡 Embedding clustering (threshold={threshold:.3f}) AUCs:")
    print(aucs_embed)

    print("🎉 Pipeline completed successfully! ✅✨")
