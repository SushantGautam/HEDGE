from algorithms import sentence_semantic_entropy, radflag, vase
import hashlib
import torch
import albumentations as A
import random
import cv2
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os, re
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import json
import shutil
from joblib import Parallel, delayed
import io
import pandas as pd
import tempfile
import asyncio

dummy_embedding_method = lambda x:  torch.randn(768, generator=torch.Generator().manual_seed(int(hashlib.sha256(x.encode()).hexdigest(), 16) % (2**32)))

class DummyNLI:
    def __init__(self): self.model=type('',(),{'config':type('',(),{'label2id':{'CONTRADICTION':0,'NEUTRAL':1,'ENTAILMENT':2}})()})()
    def __call__(self,x,batch_size=64,**kw):
        L=list(self.model.config.label2id)
        return [[{'label':l,'score':1.0 if i==int(hashlib.sha256(json.dumps(xi,sort_keys=True).encode()).hexdigest(),16)%len(L)else 0.0}for i,l in enumerate(L)]for xi in x]


def get_embeddings_batch(texts, model_name="all-MiniLM-L6-v2", batch_size=16):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    uniq = list(set(texts))
    emb = model.encode(uniq, convert_to_tensor=True, batch_size=batch_size)
    m = dict(zip(uniq, emb))
    return [m[t].cpu() for t in texts]


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
    pat_num = re.compile(r"\.completed_(\d+)\.json$"); pat_img = re.compile(r"distorted_(\d+)\.png")

    root = (Path(HEDGE_cache_path) / "datasets" / _s(str(dataset_id))).resolve(); root.mkdir(parents=True, exist_ok=True)

    def _max_meta(r):
        metas = [(int(m.group(1)), p) for p in r.glob(".completed_*.json") if (m:=pat_num.search(p.name))]
        return (max(metas, key=lambda x: x[0]) if metas else (0, None))

    def _cleanup_metas(keep):
        for p in root.glob(".completed_*.json"):
            if p != keep: p.unlink(missing_ok=True)

    def entry(m):
        d = root / m["img_name"]
        dist = sorted((p.as_posix() for p in d.glob("distorted_*.png")), key=lambda x: int(pat_img.findall(x)[0]) if pat_img.findall(x) else -1)
        return {"idx": m["idx"], "image_path": (d / "original.png").as_posix(), "question": m["question"], "answer": m["answer"], "distorted_image_paths": dist}

    Kmax, meta_path = _max_meta(root)

    # READ-ONLY
    if vqa_dict is None:
        if Kmax and num_samples <= Kmax:
            meta = json.loads(meta_path.read_text()); print(f"📂 Loaded {len(meta)} from {meta_path.name} (Kmax={Kmax}, req={num_samples})")
            return [entry(m) for m in meta]
        msg = "❌ Need more distortions than available and no data to generate.\n"
        if Kmax: msg += f"🗂️ Available for {dataset_id}: {Kmax}. Lower request ≤ {Kmax} or pass vqa_dict."
        else: msg += "🗂️ No cache yet. Pass vqa_dict to generate."
        print(msg); raise SystemExit

    # RESET if forced
    if force_regenerate and root.exists():
        print(f"⚠️ force_regenerate → rm -r {root}"); shutil.rmtree(root); root.mkdir(parents=True, exist_ok=True); Kmax, meta_path = 0, None

    # If enough already, just load existing highest and ensure it's the single one
    if Kmax and num_samples <= Kmax and not force_regenerate:
        meta = json.loads(meta_path.read_text()); _cleanup_metas(meta_path)
        print(f"⏭️ Using cache (Kmax={Kmax} ≥ req={num_samples}); meta: {meta_path.name}")
        return [entry(m) for m in meta]

    # Build unique map
    unique, idx2name = {}, []
    for e in vqa_dict:
        n = f"img_{_hash(e['image'])}"; idx2name.append(n); unique.setdefault(n, e["image"])

    def process(n, im):
        d = root / n; d.mkdir(parents=True, exist_ok=True)
        o = d / "original.png"
        if not o.exists(): im.save(o)
        w, h = im.size
        cur = sum(1 for _ in d.glob("distorted_*.png"))
        target = max(cur, Kmax)  # current max per-image in folder; we will top-up to num_samples
        for k in range(min(target, num_samples), num_samples):
            out = d / f"distorted_{k}.png"
            if not out.exists():
                arr = np.array(im); r = distort_image(h, w)(image=arr)["image"]; Image.fromarray(r).save(out)

    print(f"⚙️ Topping up distortions to {num_samples} (was Kmax={Kmax}); n={len(unique)}, n_jobs={n_jobs}")
    Parallel(n_jobs=n_jobs)(delayed(process)(n, im) for n, im in tqdm(unique.items()))

    # Write fresh SINGLE meta at the highest (requested) k and delete others
    meta = [{"idx": i, "img_name": idx2name[i], "question": e["question"], "answer": e["answer"]} for i, e in enumerate(vqa_dict)]
    new_meta = root / f".completed_{num_samples}.json"; new_meta.write_text(json.dumps(meta))
    _cleanup_metas(new_meta)
    print(f"✅ Done: → {new_meta} (single meta maintained)")
    return [entry(m) for m in meta]



def generate_answers(vqa_rad_test, n_samples=20, min_temp=0.1, max_temp=1.0, prompt_variants=None):
    from tests.medgemma import infer_batched as infer_fn

    # 1) Build the base once
    df_base = pd.DataFrame(
        [{"idx_img": s["idx"], "question": s["question"], "image": s["image_path"], "is_original": True} for s in vqa_rad_test] +
        [{"idx_img": s["idx"], "question": s["question"], "image": img, "is_original": False}
         for s in vqa_rad_test for img in s["distorted_image_paths"]]
    ).assign(temp=lambda d: d.is_original.map({True: 0.0, False: 1.0}))

    df_input_base = pd.concat(
        [df_base,
         pd.concat([df_base[df_base.is_original]] * n_samples, ignore_index=True).assign(temp=1.0)],
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
    
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args([
        "--model", model,
        "-i", input_file,
        "-o", output_file,
        "--allowed-local-media-path", allowed_media,
        "--trust-remote-code",
        "--limit-mm-per-prompt", '{"image":1,"video":0}',
        "--dtype", "auto",
        "--max-logprobs", "1",  # cap on logprobs entries
        "--logprobs-mode", "raw_logprobs",  # normalized log-probabilities
    ])

    setattr(args, "disable_frontend_multiprocessing", False)
    for k, v in extra_cli_args.items():
        setattr(args, k, v)
    await run_batch_main(args)

def run_vllm_batch_from_list(model, inputs, allowed_media):
    # Create a temporary directory to hold both input/output files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input.jsonl")
        output_file = os.path.join(tmpdir, "output.jsonl")
        with open(input_file, "w", encoding="utf-8") as f_in:
            for item in inputs:
                f_in.write(json.dumps(item) + "\n")
        print("Input payload written to:", input_file, "| Output will be at:", output_file, " and cleaned after use.")
        asyncio.run(run_vllm_batch(model, input_file, output_file, allowed_media))
        return [json.loads(line) for line in open(output_file)]