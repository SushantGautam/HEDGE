# from transformers import pipeline
    


if __name__ == '__main__': # guard  for multiprocessing
    import pickle, json
    from datasets import load_dataset

    PROMPT_VARIANTS = {
    "minimal-label": [
        {
            "role": "user",
            "content": """
Answer the question using the image only.
<image>
Question: {r.question}

Output only 'yes' or 'no'. """
        }
    ],
    "one-sentence": [ 
        {
            "role": "user",
            "content": """
Answer the question using the image only.

Rules:
- Start with "yes," or "no,"
- After the comma write a short reason explaining your answer in one sentence. 
- Do NOT mention coordinates

<image>
Question: {r.question}

Correct format examples:
yes, <why you think the answer is yes>.
no, <why you think the answer is no>.
"""

        }
    ],
}
    
    from hedge_bench.utils import (
        # PROMPT_VARIANTS,
        add_hallucination_labels_vllm,
        apply_nli_clustering,
        compute_roc_aucs,
        distort_and_cache_dataset,
        generate_answers,
        optimize_and_apply_embed_clustering,
        build_message_for_evaluation_general
    )
    
    print("💾 Starting to cache data for POPE dataset...")
    n_samples = 10
    # dataset = load_dataset("SushantGautam/RePOPE")["test"]
    # adversarial_dataset = dataset.filter(lambda x: x["category"] == "adversarial")

    # vqa_dict = [{"idx": int(s["id"]), "image": s["image"], "question": s["question"], "answer": s["answer"]} for i, s in enumerate(adversarial_dataset)]

    # for r in vqa_dict:
    #     obj = r["question"].replace("Is there ", "").replace(" in the image?", "")
    #     r['original_question'] = r["question"]
    #     r["question"] = f"Verify if {obj} appears in the image."
    #     r["answer"] = f"yes, {obj} appears" if r["answer"].lower() == "yes" else f"no, {obj} is not present"

    from hedge_bench.utils import distort_and_cache_dataset

    generated = distort_and_cache_dataset(
        dataset_id="lmms-lab-POPE",  # will cache with this name
        num_samples=n_samples,
        vqa_dict=None,
        force_regenerate=False,
        n_jobs=10
        )# take 10 samples only
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
    # # save as tmp_2026_pope.py.answers.pickle
    # breakpoint()
    with open("../results/caches/tmp_2026_pope.py.answers.pickle", "wb") as f:
        pickle.dump(answers, f)

    # answers = pickle.load(open("/home/sushant/D1/HEDGE/tmp_2026_pope.py.answers.clustered.pickle", "rb"))
    # import pandas as pd; (answers.drop_duplicates(['variant_name','media','question']).assign(pred=lambda df: df['original_low_temp'].apply(lambda x: 'no' if ' no ' in f" {x['ans'].lower()} " or "not present" in x['ans'].lower() or "no existence" in x['ans'].lower() else 'yes'), true=lambda df: df['true_answer'].str.lower().str.extract(r'^(yes|no)')[0]).groupby('variant_name')[['true','pred']].apply(lambda df: pd.crosstab(pd.Categorical(df['true'], ['no','yes']), pd.Categorical(df['pred'], ['no','yes']), dropna=False)))

    # print("🧠 Answers generated successfully! 💬")
    # # 3) Label hallucinations using a VLM judge
    # print("🔍 Labeling hallucinations with Qwen3")
    import pandas as pd
    # answers = pd.read_pickle("../results/caches/tmp_2026_pope.py.answers.clustered.pickle")
    answers = add_hallucination_labels_vllm(answers, model_name="Qwen/Qwen3-30B-A3B", dtype="auto", message_builder=build_message_for_evaluation_general)

    with open("../results/caches/tmp_2026_pope.py.filtered_answers_with_label.pickle", "wb") as f: pickle.dump(answers, f)
    # answers = pickle.load(open("../results/caches/tmp_2026_pope.py.filtered_answers_with_label.pickle", "rb"))

    print("🏷️ Hallucination labels added! ✅")

    # 4) Cluster by embeddings
    print("🧩 Performing embedding-based clustering...")
    answers_embed, threshold, _ = optimize_and_apply_embed_clustering(answers, metric_path=('one-sentence', 'metrics_embed', 'SE'))
    print(f"📈 Embedding clustering complete! Optimal threshold = {threshold:.3f} 🎯")
    aucs_embed = compute_roc_aucs(answers_embed)
    print(f"💡 Embedding clustering (threshold={threshold:.3f}) AUCs:")
    print(aucs_embed)

    # 5) Optionally, also try clustering with an NLI model
    print("🧮 Applying NLI-based clustering with DeBERTa...")
    from transformers import pipeline
    nli = pipeline("text-classification", model="microsoft/deberta-large-mnli", top_k=None, truncation=True, max_length=512)
    answers_clustered = apply_nli_clustering(answers_embed, nli, batch_size=64)
    print("📊 NLI clustering complete! 🧠")

    aucs_nli = compute_roc_aucs(answers_clustered)
    print("🧾 NLI clustering AUCs:")
    print(aucs_nli)

    print("🎉 Pipeline completed successfully! ✅✨")
    # answers_clustered to pickle
    with open("../results/caches/tmp_2026_pope.py.answers.clustered-v1.pickle", "wb") as f: pickle.dump(answers_clustered, f)
    # save aucs_nli
    with open("../results/caches/tmp_2026_pope.py.answers.clustered-v1.json", "w") as f: json.dump(aucs_nli, f)

    # breakpoint()
