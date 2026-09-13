from datasets import load_dataset, Dataset
import os
from huggingface_hub import HfApi

INSTRUCTION = (
    " In case there is an inconsistency between the context and the image content, "
    "you should follow the image. "
)

import requests

api = HfApi()
train = {f.path.split("/")[-1] for f in api.list_repo_tree("AIMClab-RUC/PhD", repo_type="dataset", path_in_repo="images/train2014")}
val = {f.path.split("/")[-1] for f in api.list_repo_tree("AIMClab-RUC/PhD", repo_type="dataset", path_in_repo="images/val2014")}

def resolve_coco_image(image_id):
    coco = str(image_id).zfill(12)
    for split, files in [("train2014", train), ("val2014", val)]:
        name = f"COCO_{split}_{coco}.jpg"
        if name in files:
            return f"https://huggingface.co/datasets/AIMClab-RUC/PhD/resolve/main/images/{split}/{name}"
    print(f"Warning: Image ID {image_id} not found in either train or val splits.")
    return None

def build_question(sample: dict, mode: str, qkey: str) -> str:
    if mode == "base":
        return sample[qkey]
    if mode == "sec":
        return sample["context"]["sec"] + INSTRUCTION + sample[qkey]
    if mode == "icc":
        return sample["context"]["icc"] + INSTRUCTION + sample[qkey]
    if mode == "ccs":
        return sample[qkey]
    raise ValueError(f"Unknown mode: {mode}")

def build_context_for_judge(sample: dict, mode: str) -> str:
    parts = []

    if sample.get("subject"):
        parts.append(f'The questioned subject is "{sample["subject"]}".')
    if sample.get("hitem"):
        parts.append(f'The posible hallucination can be like "{sample["hitem"]}".')
    if sample.get("gt"):
        parts.append(f'The ground truth is "{sample["gt"]}".')
    if mode == "ccs" and sample.get("ccs_description"):
        parts.append(
            f'The image is counter-common-sense: "{sample["ccs_description"]}".'
        )

    return " ".join(parts)

def sample_to_vqa6_rows(sample: dict, images_root: str = "images") -> list[dict]:
    rows = []

    is_ccs = bool(sample.get("ccs_description"))
    image_id = sample["image_id"]

    if is_ccs:
        modes = ["ccs"]
        # CCS images are stored separately and are already available via a stable path.
        image_path = os.path.join(
            images_root, "CCS_images", f"{image_id}.png"
        )
    else:
        modes = ["base", "sec", "icc"]
        # Defer resolving COCO image URLs until after we sample the subset.
        image_path = None

    for mode in modes:
        for qkey, answer in [("yes_question", "yes"), ("no_question", "no")]:
            rows.append(
                {
                    "task": sample["task"],
                    "modes": mode,
                    "image": image_path,
                    "image_id": image_id,
                    "question": build_question(sample, mode, qkey),
                    "answer": answer,
                    "context_for_judge": build_context_for_judge(sample, mode),
                }
            )

    return rows

def convert_hf_phd_to_vqa6(images_root: str = "images") -> Dataset:
    ds = load_dataset("AIMClab-RUC/PhD", split="test")

    all_rows = []
    for sample in ds:
        all_rows.extend(sample_to_vqa6_rows(sample, images_root=images_root))

    vqa6 = Dataset.from_list(all_rows)
    return vqa6

# usage
vqa6 = convert_hf_phd_to_vqa6(
    images_root="https://huggingface.co/datasets/AIMClab-RUC/PhD/resolve/main/images"
).to_pandas()

# Sample a subset first, then resolve COCO image paths only for the selected rows.
vqa6_top_df = (
    vqa6.groupby(["task", "modes"], group_keys=False)
    .apply(lambda x: x.sample(n=min(500, len(x)), random_state=42))
    .reset_index(drop=True)
)

# Resolve COCO images lazily so we don't download every image for the full dataset.
def _resolve_image_path(row):
    if row["modes"] == "ccs":
        return row["image"]
    return resolve_coco_image(row["image_id"])

from tqdm import tqdm

tqdm.pandas()

vqa6_top_df["image"] = vqa6_top_df.progress_apply(_resolve_image_path, axis=1)
vqa6_top = Dataset.from_pandas(vqa6_top_df, preserve_index=False)

from datasets import Image
vqa6_top = vqa6_top.cast_column("image", Image(decode=True))

from datasets import DatasetDict
DatasetDict({"test": vqa6_top}).push_to_hub( "SushantGautam/AIMClab-RUC_PhD_subset" )

print(vqa6_top)
print(vqa6_top[0])
breakpoint()