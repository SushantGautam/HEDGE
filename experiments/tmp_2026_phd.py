
if __name__ == '__main__': # guard  for multiprocessing
    from datasets import load_dataset
    import json
    ds = load_dataset("SushantGautam/AIMClab-RUC_PhD_subset", split="test")
    vqa_dict = [{"idx": i, "image": s["image"], "question": s["question"], "answer": s["answer"], "category": s["task"], "modes": s['modes'], 
    "description": s["context_for_judge"].replace("can be like", "can be about") 
    } for i, s in enumerate(ds)]
# 
    vqa_all_dict ={i['idx']: i for i in vqa_dict}

    PROMPT_VARIANTS = {
    "minimal-label": [
        {
            "role": "user",
            "content": """
<image>
Question: {r.question}

Output only 'yes' or 'no'. """
        }
    ],
    "one-sentence": [ 
        {
            "role": "user",
            "content": """
<image>
Question: {r.question}

Answer Rules:
- Start with "yes," or "no,"
- After the comma write a short reason explaining your answer in one sentence. 
- Do NOT mention coordinates

Correct answer format examples:
yes, <why you think the answer is yes>.
no, <why you think the answer is no>.
"""

        }
    ],
}
    import pickle

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
    
    print("💾 Starting to cache data for PHD dataset...")
    n_samples = 10
    # breakpoint()

    # from hedge_bench.utils import distort_and_cache_dataset

    # generated = distort_and_cache_dataset(
    #     dataset_id="lmms-lab-PHD",  # will cache with this name
    #     num_samples=n_samples,
    #     vqa_dict=None,
    #     force_regenerate=False,
    #     n_jobs=10
    #     )# take 10 samples only

    # import random; random.seed(42); generated = random.sample(generated, 500)

    # for idx, row in enumerate(generated):
    #     row['description'] = vqa_all_dict[idx]['description']

    print("✅ Dataset prepared successfully! 📊")

    # 2) Sample answers from a vision-language model
    # print("🤖 Generating answers ...")
    # answers = generate_answers(
    #     generated,
    #     n_answers_high=n_samples,
    #     min_temp=0.1,
    #     max_temp=1.0,
    #     prompt_variants=PROMPT_VARIANTS,
    #     model="Qwen/Qwen2-VL-2B-Instruct",
    #     extra_cli_args={"dtype":"auto"},
    # )
    # save as tmp_2026_phd.py.answers.pickle
    # with open("../results/redo_2026/tmp_2026_phd.py.answers.pickle", "wb") as f:
    #     pickle.dump(answers, f)
    # import pickle
    # answers = pickle.load(open("/home/sushant/D1/HEDGE/tmp_2026_phd.py.filtered_answers_with_label_no_desc.pickle", "rb"))
    # for index, row in answers.iterrows():
    #     answers.loc[index, 'description'] = None


    # import pandas as pd; (answers.drop_duplicates(['variant_name','media','question']).assign(pred=lambda df: df['original_low_temp'].apply(lambda x: 'no' if ' no ' in f" {x['ans'].lower()} " or "not present" in x['ans'].lower() or "no existence" in x['ans'].lower() else 'yes'), true=lambda df: df['true_answer'].str.lower().str.extract(r'^(yes|no)')[0]).groupby('variant_name')[['true','pred']].apply(lambda df: pd.crosstab(pd.Categorical(df['true'], ['no','yes']), pd.Categorical(df['pred'], ['no','yes']), dropna=False)))
    # print("🧠 Answers generated successfully! 💬")
    # 3) Label hallucinations using a VLM judge
    # print("🔍 Labeling hallucinations with Qwen3")
    # answers = add_hallucination_labels_vllm(answers, model_name="Qwen/Qwen3-30B-A3B", dtype="auto", message_builder=build_message_for_evaluation_general)
    # use Qwen/Qwen3-30B-A3B in real
    # breakpoint()
    # with open("../results/redo_2026/tmp_2026_phd.py.filtered_answers_with_label_no_desc.pickle", "wb") as f: pickle.dump(answers, f)
    # import pickle
    # answers = pickle.load(open("../results/redo_2026/tmp_2026_phd.py.filtered_answers_with_label.pickle", "rb"))
    # answers['task'] = answers['idx_img'].map(lambda x: vqa_all_dict[x]['category'])
    # answers['modes'] = answers['idx_img'].map(lambda x: vqa_all_dict[x]['modes'])
    # answers = pickle.load(open("/home/sushant/D1/HEDGE/tmp_2026_phd.py.filtered_answers_with_label_no_desc.pickle", "rb"))
    # print("🏷️ Hallucination labels added! ✅")
    
    # mask = answers['hallucination_label'] != -1
    # removed_rows = (~mask).sum()
    # answers = answers.loc[mask].reset_index(drop=True)
    # print(f"Rows removed: {removed_rows}")
    # print(f"Remaining rows: {len(answers)}")
    
    # # breakpoint()
    # # 4) Cluster by embeddings
    answers = pickle.load(open("../results/redo_2026/tmp_2026_phd.py.answers.clustered-no_description.pickle", "rb"))

    print("🧩 Performing embedding-based clustering...")
    answers_embed, threshold, _ = optimize_and_apply_embed_clustering(answers, metric_path=('minimal-label', 'metrics_embed', 'VASE'))
    print(f"📈 Embedding clustering complete! Optimal threshold = {threshold:.3f} 🎯")

    aucs_embed = compute_roc_aucs(answers_embed)
    print(f"💡 Embedding clustering (threshold={threshold:.3f}) AUCs:")
    print(aucs_embed)
    breakpoint()

    # 5) Optionally, also try clustering with an NLI model
        # with open("../results/redo_2026/tmp_2026_phd.py.answers.clustered-no_description.pickle", "wb") as f: pickle.dump(answers_clustered, f)
    print("🧮 Applying NLI-based clustering with DeBERTa...")
    from transformers import pipeline
    nli = pipeline("text-classification", model="microsoft/deberta-large-mnli", top_k=None, truncation=True, max_length=512)
    answers_clustered = apply_nli_clustering(answers_embed, nli, batch_size=64)
    print("📊 NLI clustering complete! 🧠")

    aucs_nli = compute_roc_aucs(answers_clustered)
    print("🧾 NLI clustering AUCs:")
    print(aucs_nli)
    aucs_nli = compute_roc_aucs(answers_clustered)
    print("🎉 Pipeline completed successfully! ✅✨")

    # answers_clustered to pickle
    # with open("../results/redo_2026/tmp_2026_phd.py.answers.clustered-no_description.pickle", "wb") as f: pickle.dump(answers_clustered, f)
    # # save aucs_nli
    # with open("../results/redo_2026/tmp_2026_phd.py.answers.clustered-no_description.json", "w") as f: json.dump(aucs_nli, f)
    # # breakpoint()
    
    answers_clustered['task'] = answers_clustered['idx_img'].map(lambda x: vqa_all_dict[x]['category'])
    answers_clustered['modes'] = answers_clustered['idx_img'].map(lambda x: vqa_all_dict[x]['modes'])
    # # breakpoint()
    # answers_clustered = pickle.load(open("/home/sushant/D1/HEDGE/tmp_2026_phd.py.answers.clustered-description.pickle", "rb"))
    breakpoint()


    # { "variant_name/modes": { k: next(iter(compute_roc_aucs(g).values())) for k, g in answers_clustered.groupby(["variant_name", "modes"]) }, "variant_name/task": { k: next(iter(compute_roc_aucs(g).values())) for k, g in answers_clustered.groupby(["variant_name", "task"]) }, }
    
