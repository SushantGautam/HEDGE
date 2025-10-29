from hedge_bench.algorithms import sentence_semantic_entropy, radflag, vase
import hashlib
import torch
import albumentations as A
import random
import cv2
import re
from datasets import load_dataset
import numpy as np
import os, re
from PIL import Image
from pathlib import Path
import json
import shutil
from joblib import Parallel, delayed
import io
import pandas as pd
import tempfile
import optuna
import asyncio
from hedge_bench.algorithms import cluster_terms_by_embedding, cluster_terms_by_nli, get_nli_labels, cluster_from_nli_labels
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from functools import reduce
import operator

tqdm.pandas() 





dummy_embedding_method = lambda x:  torch.randn(768, generator=torch.Generator().manual_seed(int(hashlib.sha256(x.encode()).hexdigest(), 16) % (2**32)))

class DummyNLI:
    def __init__(self,seed=42): torch.manual_seed(seed);self.model=type('',(),{'config':type('',(),{'label2id':{'CONTRADICTION':0,'NEUTRAL':1,'ENTAILMENT':2}})()})()
    def __call__(self,x,batch_size=64, *args, **kwargs):
        return [[{'label':l,'score':1.0 if i==torch.randint(0,len(self.model.config.label2id),(1,),generator=torch.Generator().manual_seed(int(hashlib.sha256(str(s).encode()).hexdigest(),16)%(2**32))).item()else 0.0}for i,l in enumerate(self.model.config.label2id)]for s in x]

def get_embeddings_batch(texts, model_name="all-MiniLM-L6-v2", batch_size=16):
    from sentence_transformers import SentenceTransformer
    uniq = list(dict.fromkeys(texts))
    model = SentenceTransformer(model_name)
    uniq_emb = model.encode(uniq, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)
    lookup = {u: e for u, e in zip(uniq, uniq_emb)}
    out = [lookup[t].cpu() for t in texts]
    return out

def compute_metrics(n, cluster_ids, normal, noisy, normal_logs, noisy_logs, alpha):
    ent_clean, dist_clean = sentence_semantic_entropy(normal_logs, cluster_ids[1:1+n])
    ent_noisy, dist_noisy = sentence_semantic_entropy(noisy_logs, cluster_ids[1+n:])
    return {
        "SE": float(ent_clean),
        "RadFlag": radflag(cluster_ids, n),
        "VASE": vase(n, cluster_ids, dist_clean, dist_noisy, alpha)
    }


def distort_image(h, w):
    return A.Compose([
        # --- Safe geometry ---
        A.Affine(
            rotate=random.choice([(-10, -2), (2, 10)]),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            fit_output=True,
            border_mode=cv2.BORDER_CONSTANT,
        ),

        # --- Color jitter ---
        A.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.95, 1.05),
            hue=(-0.02, 0.02),
        ),
        # --- Noise ---
        A.GaussNoise(std_range=(0.07, 0.07), mean_range=(0.0, 0.0), p=1.0),
        A.ShotNoise(scale_range=(0.014, 0.014), p=1.0)

    ]
  )



HEDGE_cache_path= ".cache_HEDGE/"

def generate_and_cache_dataset(dataset_id, vqa_dict=None, num_samples=10, n_jobs=20, force_regenerate=False):
    assert dataset_id and str(dataset_id).strip(), "❌ 'dataset_id' must be provided and non-empty."
    def _s(x): return re.sub(r'[^a-zA-Z0-9_-]', '_', x.strip())
    def _hash(img): return hashlib.md5(np.array(img).tobytes()).hexdigest()
    pat_num = re.compile(r"\.completed_(\d+)\.json$")
    pat_img = re.compile(r"distorted_(\d+)\.png")

    root = (Path(HEDGE_cache_path) / "datasets" / _s(str(dataset_id))).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def _max_meta(r):
        metas = [(int(m.group(1)), p) for p in r.glob(".completed_*.json") if (m := pat_num.search(p.name))]
        return (max(metas, key=lambda x: x[0]) if metas else (0, None))

    def _cleanup_metas(keep):
        for p in root.glob(".completed_*.json"):
            if p != keep:
                p.unlink(missing_ok=True)

    def entry(m):
        d = root / m["img_name"]
        dist_all = sorted(
            (p.as_posix() for p in d.glob("distorted_*.png")),
            key=lambda x: int(pat_img.findall(x)[0]) if pat_img.findall(x) else -1
        )
        dist = dist_all[:num_samples]  # ✅ only cap number of distorted images returned
        return {
            "idx": m["idx"],
            "image_path": (d / "original.png").as_posix(),
            "question": m["question"],
            "answer": m["answer"],
            "description": m.get("description", None),
            "distorted_image_paths": dist
        }

    Kmax, meta_path = _max_meta(root)

    # READ-ONLY
    if vqa_dict is None:
        if Kmax and num_samples <= Kmax:
            meta = json.loads(meta_path.read_text())
            print(f"📂 Loaded {len(meta)} from {meta_path.name} (Kmax={Kmax}, req={num_samples})")
            return [entry(m) for m in meta]
        msg = "❌ Need more distortions than available and no data to generate.\n"
        if Kmax:
            msg += f"🗂️ Available for {dataset_id}: {Kmax}. Lower request ≤ {Kmax} or pass vqa_dict."
        else:
            msg += "🗂️ No cache yet. Pass vqa_dict to generate."
        print(msg)
        raise SystemExit

    # RESET if forced
    if force_regenerate and root.exists():
        print(f"⚠️ force_regenerate → rm -r {root}")
        shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        Kmax, meta_path = 0, None

    # If enough already, just load existing highest and ensure it's the single one
    if Kmax and num_samples <= Kmax and not force_regenerate:
        meta = json.loads(meta_path.read_text())
        _cleanup_metas(meta_path)
        print(f"⏭️ Using cache (Kmax={Kmax} ≥ req={num_samples}); meta: {meta_path.name}")
        return [entry(m) for m in meta]

    # Build unique map
    unique, idx2name = {}, []
    for e in vqa_dict:
        n = f"img_{_hash(e['image'])}"
        idx2name.append(n)
        unique.setdefault(n, e["image"])

    def process(n, im):
        d = root / n
        d.mkdir(parents=True, exist_ok=True)
        o = d / "original.png"
        if not o.exists():
            im.save(o)
        w, h = im.size
        cur = sum(1 for _ in d.glob("distorted_*.png"))
        target = max(cur, Kmax)
        for k in range(min(target, num_samples), num_samples):
            out = d / f"distorted_{k}.png"
            if not out.exists():
                arr = np.array(im)
                r = distort_image(h, w)(image=arr)["image"]
                Image.fromarray(r).save(out)

    print(f"⚙️ Topping up distortions to {num_samples} (was Kmax={Kmax}); n={len(unique)}, n_jobs={n_jobs}")
    Parallel(n_jobs=n_jobs)(delayed(process)(n, im) for n, im in tqdm(unique.items()))

    # Write fresh SINGLE meta at the highest (requested) k and delete others
    meta = [{"idx": i, "img_name": idx2name[i], "question": e["question"], "answer": e["answer"], "description": e.get("description", None)}
            for i, e in enumerate(vqa_dict)]
    new_meta = root / f".completed_{num_samples}.json"
    new_meta.write_text(json.dumps(meta))
    _cleanup_metas(new_meta)
    print(f"✅ Done: → {new_meta} (single meta maintained)")
    return [entry(m) for m in meta]

# old method ,. .  use ms-swift 
def generate_answers_swift(vqa_rad_test, n_answers_high=20, min_temp=0.1, max_temp=1.0, prompt_variants=None):
    from hedge_bench.tests.medgemma import infer_batched as infer_fn

    # 1) Build the base once
    df_base = pd.DataFrame(
        [{"idx_img": s["idx"], "question": s["question"], "image": s["image_path"], "is_original": True} for s in vqa_rad_test] +
        [{"idx_img": s["idx"], "question": s["question"], "image": img, "is_original": False}
         for s in vqa_rad_test for img in s["distorted_image_paths"]]
    ).assign(temp=lambda d: d.is_original.map({True: 0.0, False: 1.0}))

    df_input_base = pd.concat(
        [df_base,
         pd.concat([df_base[df_base.is_original]] * n_answers_high, ignore_index=True).assign(temp=1.0)],
        ignore_index=True
    ).reset_index(drop=True)

    # 2) Copy per variant (cheap) and tag with variant_name
    variant_names = list(prompt_variants.keys())
    df_input = pd.concat(
        [df_input_base.assign(variant_name=vn) for vn in variant_names],
        ignore_index=True
    )

    # 3) Build vqa_rad_data once for all rows, selecting messages by variant
    vqa_rad_data = df_input.apply(
        lambda r: {
            "messages": [{**m, "content": m["content"].format(r=r)} for m in prompt_variants[r.variant_name]],
            "images": [r.image],
            "temperature": min_temp if r.temp == 0 else max_temp,
        },
        axis=1
    ).tolist()
    
    # 4) Single inference call
    answers, logprobs = infer_fn(vqa_rad_data, temperature=1.0)
    df_input["answer"], df_input["logprobs"] = answers, logprobs

    # 5) Collapse logic unchanged
    def collapse(g):
        f = lambda cond: [{"ans": a, "logprob": lp} for a, lp in zip(g.loc[cond, "answer"], g.loc[cond, "logprobs"])]
        return pd.Series({
            "idx_img": g.idx_img.iloc[0],
            "question": g.loc[g.is_original, "question"].iloc[0],
            "image": g.loc[g.is_original, "image"].iloc[0],
            "original_high_temp": f(g.is_original & (g.temp == 1)),
            "distorted_high_temp": f(~g.is_original & (g.temp == 1)),
            "original_low_temp": f(g.is_original & (g.temp == 0))[0],
            "variant_name": variant_name,
        })

    # 6) Produce per-variant outputs
    all_dfs = {}
    for variant_name, g_variant in df_input.groupby("variant_name"):
        all_dfs[variant_name] = (
            g_variant.groupby("idx_img", as_index=False)
            .apply(collapse)
            .reset_index(drop=True)
        )

    return pd.concat(all_dfs.values(), ignore_index=True)



async def run_vllm_batch(model, input_file, output_file, allowed_media, extra_cli_args={}): 
    from vllm.utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.run_batch import make_arg_parser, main as run_batch_main
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args([
        "--model", model,
        "-i", input_file,
        "-o", output_file,
        "--allowed-local-media-path", allowed_media,
        "--trust-remote-code",
        "--limit-mm-per-prompt", '{"image":1,"video":0}',
        "--max-model-len", "10000",
        "--dtype", "auto",
        "--max-logprobs", "1",  # cap on logprobs entries
        "--logprobs-mode", "raw_logprobs",  # normalized log-probabilities
    ])

    setattr(args, "disable_frontend_multiprocessing", False)
    for k, v in extra_cli_args.items():
        setattr(args, k, v)
    await run_batch_main(args)

def run_vllm_batch_from_list(model, inputs, allowed_media=None, extra_cli_args={}):
    # Create a temporary directory to hold both input/output files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input.jsonl")
        output_file = os.path.join(tmpdir, "output.jsonl")
        with open(input_file, "w", encoding="utf-8") as f_in:
            for item in inputs:
                f_in.write(json.dumps(item) + "\n")
        print("Input payload written to:", input_file, "| Output will be at:", output_file, " and cleaned after use.")
        asyncio.run(run_vllm_batch(model, input_file, output_file, allowed_media or "", extra_cli_args=extra_cli_args))
        return [json.loads(line) for line in open(output_file)]

def make_seq_for_clustering(row, alpha=1.0, append_question=False):
    p = (row["question"] + " ") if append_question else ""
    normal = [p + d["ans"] for d in row.original_high_temp]
    noisy = [p + d["ans"] for d in row.distorted_high_temp][:20]
    logn = [np.mean(d["logprob"]) for d in row.original_high_temp]
    logd = [np.mean(d["logprob"]) for d in row.distorted_high_temp][:20]
    seq = [p + row.original_low_temp['ans']] + normal + noisy
    n = len(normal)
    return {"n": n, "seq_input": seq, "normal": normal, "noisy": noisy, "logn": logn, "logd": logd, "alpha": alpha}


def apply_nli_clustering(dataframe, nli_model, batch_size=1024, append_question=False):
    dataframe = dataframe.copy()
    dataframe["clustering_input"] = dataframe.progress_apply(make_seq_for_clustering, append_question=append_question,  axis=1)
    all_sequences = [x["seq_input"] for x in dataframe["clustering_input"]]
    nli_labels = get_nli_labels(all_sequences, nli_model, B=batch_size)
    clusters = cluster_from_nli_labels(nli_labels)
    dataframe["cluster_nli"] = clusters
    dataframe["metrics_nli"] = dataframe.apply(
        lambda row: compute_metrics(
            n=row["clustering_input"]["n"],
            cluster_ids=row["cluster_nli"],
            normal=row["clustering_input"]["normal"],
            noisy=row["clustering_input"]["noisy"],
            normal_logs=row["clustering_input"]["logn"],
            noisy_logs=row["clustering_input"]["logd"],
            alpha=row["clustering_input"]["alpha"],
        ),
        axis=1,
    )
    dataframe = dataframe.drop(columns=["clustering_input"])
    return dataframe


def apply_embed_clustering(dataframex, embedding_cached_fn, threshold=0.90, append_question=False):
    dataframe = dataframex.copy()
    dataframe["clustering_input"] = dataframe.progress_apply(make_seq_for_clustering, append_question=append_question,  axis=1)
    all_sequences = [x["seq_input"] for x in dataframe["clustering_input"]]
    ids_embds = []
    for seq in tqdm(all_sequences, desc="Clustering all sequences by embedding", unit="sequence"):
        ids_embd = cluster_terms_by_embedding(seq, embedding_cached_fn, threshold=threshold)
        ids_embds.append(ids_embd)

    dataframe["cluster_embed"] = ids_embds
    dataframe["metrics_embed"] = dataframe.apply(
        lambda row: compute_metrics(
            n=row["clustering_input"]["n"],
            cluster_ids=row["cluster_embed"],
            normal=row["clustering_input"]["normal"],
            noisy=row["clustering_input"]["noisy"],
            normal_logs=row["clustering_input"]["logn"],
            noisy_logs=row["clustering_input"]["logd"],
            alpha=row["clustering_input"]["alpha"],
        ),
        axis=1,
    )
    dataframe = dataframe.drop(columns=["clustering_input"])
    return dataframe


evaluator_struct_output_schema={"type":"object","properties":{"reason":{"type":"string","description":"One short sentence (≤20 words) explaining why the generated_answer matches or doesn’t match the correct_answer."},"score":{"type":"integer","enum":[0,1],"description":"1 if semantically equivalent, 0 otherwise."}},"required":["reason","score"]}

def build_message_for_evaluation(item, add_description=True):
    system_msg = """
    You are a strict medical evaluator.

    You will be given these inputs:
    - question: the question asked about the medical image
    """
    if add_description and item.get("description", None):
        system_msg += """
    - description: clarifies what the question expects 
    """

    system_msg += """
    - correct_answer: the clinically verified correct answer
    - generated_answer: the answer produced by the vision-language model (VLM)

    Your task:

    - Compare the generated_answer with the correct_answer **only on clinical and semantic meaning.**
    - Score as 0 **only if the generated_answer introduces false, contradictory, or medically incorrect information** compared to the correct_answer.
    - Minor wording or phrasing differences are acceptable if meaning is equivalent.

    Output format (STRICT JSON):

    ```json
    {"reason": "<one concise sentence (≤20 words)>", "score": 1 or 0}
    ```

    where:
      - "reason": briefly explains why the answer matches or contradicts the correct answer.
      - "score": 1 if clinically and semantically equivalent,
                 0 if contradictory or factually incorrect.

    Do not include any commentary or text outside the JSON block.
    """

    user_msg = f"""
    question: {item['question']}
    """
    if add_description and item.get("description", None):
        user_msg += f"""
    description: {item['description']}
    """
    user_msg += f"""
    correct_answer: {item['true_answer']}
    generated_answer: {item['original_low_temp']['ans']}
    """

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg + " /no_think"}
    ]




def parse_vllm_outputs_from_evaluator(outputs):
    print(f"Got {len(outputs)} outputs from vllm")
    parsed_results=[]
    for o in outputs:
        i=int((o.get("custom_id","unknown")).split("-")[-1])
        try:
            j=json.loads(re.findall(r'\{.*?\}',o.get("response",{}).get("body",{}).get("choices",[{}])[0].get("message",{}).get("content",""),re.S)[0])
            score=int(j.get("score",-1))
            assert score in [0,1]
        except Exception as e:
            print(f"⚠️ Failed parsing for {o.get('custom_id')}: {e}")
            score=-1
        parsed_results.append({"custom_id":o.get("custom_id"),"idx":i,"result":score})
    return {r["custom_id"]:r["result"] for r in parsed_results}


def compute_roc_aucs(df):
    assert "hallucination_label" in df.columns, "DataFrame must contain 'hallucination_label' column."
    aucs = {}
    for variant_name, group in df.groupby("variant_name"):
        aucs[variant_name] = {}
        for mcol in [c for c in group.columns if c.startswith("metrics")]:
            metrics_df = pd.json_normalize(group[mcol])
            aucs[variant_name][mcol] = {
                k: roc_auc_score(
                    group.hallucination_label,
                    v if k == "RadFlag" else 1 - v
                )
                for k, v in metrics_df.items()
            }
    return aucs

def optimized_cluster_threshold(df_vqa_rad, embedding_cached_fn, metric_path=('default', 'metrics_embed', 'SE'),
                       threshold_range=(0.8, 0.99), n_trials=20, debug=False, append_question=False):
    history = {}
    def func_to_optimize(threshold):
        t = round(float(threshold), 6)
        if t in history:
            res = history[t]
        else:
            res = compute_roc_aucs(apply_embed_clustering(df_vqa_rad, embedding_cached_fn, threshold=t, append_question=append_question))
            history[t] = res
        if debug:
            print(f"t={t} -> {res}")
        return reduce(operator.getitem, metric_path, res)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda tr: func_to_optimize(tr.suggest_float("t", *threshold_range)), n_trials=n_trials)
    best_t = study.best_params["t"]
    best_score = func_to_optimize(best_t)
    print(f"best_t={best_t:.4f}, metric={best_score:.6f}")
    return best_t, dict(sorted(history.items()))


PROMPT_VARIANTS = {
    "default": [
        {"role": "system", "content": "You are a medical image analysis expert."},
        {"role": "user", "content": "<image> Answer this question as concisely as possible based on the provided image: {r.question}"},
    ],
    "Concise": [
        {
            "role": "system",
            "content": (
                "You are a medical vision-language assistant. "
                "Given a medical image and a clinical question, provide only the minimal, clinically correct answer. "
                "Answers may be 'yes', 'no', or a short medically relevant label (e.g., modality, anatomy, finding). "
                "Do not generate full sentences, explanations, or extra text—only output exactly the expected label."
            ),
        },
        {"role": "user", "content": "Question: {r.question}"},
    ],
    "SingleSentence": [
        {"role": "system", "content": "You are a medical image analysis expert; always respond in no more than a single sentence."},
        {"role": "user", "content": "<image> Answer this question as concisely as possible based on the provided image: {r.question}"},
    ],
    "ShortClinical": [
        {
            "role": "system",
            "content": (
                "You are a medical image analysis expert. "
                "Given an image and a question, provide a concise and accurate response. "
                "The answer should be a short phrase, slightly longer than a single label if needed, "
                "but not a full grammatical sentence and without a period at the end. "
                "Avoid explanations or extra text—only give the short direct answer."
            ),
        },
        {"role": "user", "content": "Question: {r.question}"},
    ],
}

def to_openai_multimodal_payload(old, question, image_urls):
    return [
        {"role": m["role"],
         "content": (
             [{"type": "image_url", "image_url": {"url": u}} for u in image_urls] if m["role"] == "user" else []
         ) + [{"type": "text", "text": m["content"].replace("{r.question}", question)}]}
        for m in old
    ]


# new method ,. .  use vllm
def generate_answers(
    vqa_rad_test,
    n_answers_high=20,
    min_temp=0.1,
    max_temp=1.0,
    prompt_variants=None,
    model="google/medgemma-4b-it",  # change to your actual VLM
    max_completion_tokens=512,
    extra_cli_args=None):
    # 1) Build the base once
    df_base = pd.DataFrame(
        [{"idx_img": s["idx"], "question": s["question"], "image": s["image_path"], "is_original": True, "true_answer": s.get("answer"), "description": s.get("description")} for s in vqa_rad_test]
        +
        [{"idx_img": s["idx"], "question": s["question"], "image": img, "is_original": False, "true_answer": s.get("answer"), "description": s.get("description")}
         for s in vqa_rad_test for img in s["distorted_image_paths"]]
    ).assign(temp=lambda d: d.is_original.map({True: 0.0, False: 1.0}))

    # 2) Replicate originals n_answers_high times at high temperature
    df_input_base = pd.concat(
        [
            df_base,
            pd.concat([df_base[df_base.is_original]] * n_answers_high, ignore_index=True).assign(temp=1.0),
        ],
        ignore_index=True,
    ).reset_index(drop=True)

    # 3) Copy per variant and tag
    variant_names = list(prompt_variants.keys())
    df_input = pd.concat(
        [df_input_base.assign(variant_name=vn) for vn in variant_names],
        ignore_index=True
    ).reset_index(drop=True)

    # 4) Build batched HTTP-style inputs for run_vllm_batch_from_list
    
    inputs = []
    for i, row in df_input.iterrows():
        body = {
            "model": model,
            "messages": to_openai_multimodal_payload( prompt_variants[row.variant_name], row.question, [f"file://{row.image}"]),
            "max_completion_tokens": max_completion_tokens,
            "temperature": min_temp if row.temp == 0.0 else max_temp,
            "logprobs": True,
            "top_logprobs": 1,
        }

        inputs.append({
            "custom_id": f"vqa-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        })

    # 5) Run vLLM
    outputs = run_vllm_batch_from_list(
        inputs=inputs,
        model=model,
        # allow images to be sent from disk if your helper gates this
        allowed_media="/",
        extra_cli_args={**{"dtype":"auto", "tensor-parallel-size":1, "gpu-memory-utilization":0.9, "enable-prefix-caching":True, "max-model-len":2500}, **(extra_cli_args or {})}
    )
    outputs = sorted(outputs, key=lambda x: int(x["custom_id"].split('-')[-1]))

    answers = []
    logprobs_list = []
    pat = re.compile(r"(?:<\|?.+?\|?>|\[[^\]]+\])")
    for out in outputs:
        answer = out['response']['body']['choices'][0]['message']['content'].strip()
        logprobs = [t['logprob'] for t in out['response']['body']['choices'][0]['logprobs']['content'] if not pat.match(t['token'])]
        answers.append(answer)
        logprobs_list.append(logprobs)
    df_input["answer"] = answers
    df_input["logprobs"] = logprobs_list

    # 7) Collapse per (idx_img, variant_name) into your desired structure
    def collapse(g):
        def pack(cond):
            return [{"ans": a, "logprob": lp} for a, lp in zip(g.loc[cond, "answer"], g.loc[cond, "logprobs"])]

        return pd.Series({
            "idx_img": g.idx_img.iloc[0],
            "image": g.loc[g.is_original, "image"].iloc[0],
            "question": g.loc[g.is_original, "question"].iloc[0],
            "true_answer": g.loc[g.is_original, "true_answer"].iloc[0],
            "description": g.loc[g.is_original, "description"].iloc[0],
            "original_high_temp": pack(g.is_original & (g.temp == 1.0)),
            "distorted_high_temp": pack(~g.is_original & (g.temp == 1.0)),
            "original_low_temp": pack(g.is_original & (g.temp == 0.0))[0],
            "variant_name": g.variant_name.iloc[0],
        })

    all_dfs = []
    for variant_name, g_variant in df_input.groupby("variant_name"):
        all_dfs.append(
            g_variant
            .groupby("idx_img", group_keys=False)
            .apply(lambda g: collapse(g.assign(idx_img=g.name)))
            .reset_index(drop=True)
        )

    return pd.concat(all_dfs, ignore_index=True)


def add_hallucination_labels_vllm(
    dataframe,
    model_name="Qwen/Qwen3-30B-A3B",
    reasoning_parser="qwen3",
    evaluator_schema=None,
    add_description=True,
    dtype="auto",
    tp_size=1,
    gpu_mem_util=0.90,
    enable_prefix_caching=True,
    max_model_len=2500,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    allowed_media=None,
    max_completion_tokens=1000):

    assert "true_answer" in dataframe, "Column 'true_answer' not found in DataFrame"

    if evaluator_schema is None:
        evaluator_schema = evaluator_struct_output_schema

    df = dataframe.copy()

    # --- Build a stable key for deduplication
    df["_hedge_key"] = (df["image"].astype(str) + "||" +df["question"].astype(str) + "||" +df["true_answer"].astype(str))
    df_unique = df.drop_duplicates(subset=["_hedge_key"]).copy().reset_index(drop=True)
    key_to_custom_id = {k: f"hedge-{i}" for i, k in enumerate(df_unique["_hedge_key"])}
    inputs = [
        {
            "custom_id": key_to_custom_id[row["_hedge_key"]],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": build_message_for_evaluation(row, add_description=add_description),
                "max_completion_tokens": max_completion_tokens,
            },
        }
        for _, row in df_unique.iterrows()
    ]

    # --- Run the evaluator once per unique key
    outputs = run_vllm_batch_from_list(
        inputs=inputs,
        model=model_name,
        allowed_media=allowed_media,
        extra_cli_args={
            "reasoning-parser": reasoning_parser,
            "structured-outputs-config": evaluator_schema,
            "dtype": dtype,
            "tensor-parallel-size": tp_size,
            "gpu-memory-utilization": gpu_mem_util,
            "enable-prefix-caching": enable_prefix_caching,
            "max-model-len": max_model_len,
            "override-generation-config": (
                f'{{"temperature":{temperature},"top_p":{top_p},"top_k":{top_k},"min_p":{min_p}}}'
            ),
        },
    )

    id_to_score = parse_vllm_outputs_from_evaluator(outputs)
    key_to_score = {k: id_to_score[cid] for k, cid in key_to_custom_id.items()}
    df["hallucination_label"] = df["_hedge_key"].map(key_to_score)
    df.drop(columns=["_hedge_key"], inplace=True)
    return df



def optimize_and_apply_embed_clustering(
    df,
    metric_path=("default", "metrics_embed", "SE"),
    threshold_range=(0.8, 0.99),
    n_trials=20,
    batch_size=16,
    debug=False,
    embedding_cache=None,
    append_question=False
):
    assert "hallucination_label" in df.columns, "DataFrame must contain 'hallucination_label' column."
    unique_answers = list({
        a for r in df.apply(
            lambda r: [d["ans"] for d in r["original_high_temp"]]
                    + [d["ans"] for d in r["distorted_high_temp"]]
                    + [r["original_low_temp"]["ans"]],
            axis=1
        ) for a in r if isinstance(a, str)
    })

    cache = embedding_cache or {}
    to_embed = [a for a in unique_answers if a not in cache]
    if to_embed:
        cache.update(dict(zip(to_embed, get_embeddings_batch(to_embed, batch_size=batch_size))))
    embed_fn = lambda x, _c=cache: _c.get(x)

    thr, history = optimized_cluster_threshold(
        df, embed_fn, metric_path=metric_path,
        threshold_range=threshold_range, n_trials=n_trials, debug=debug, append_question=append_question
    )
    return apply_embed_clustering(df, embed_fn, threshold=thr, append_question=append_question), thr, history
