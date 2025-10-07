
import hashlib
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import torch
from algorithms import cluster_terms_by_embedding, cluster_terms_by_nli
from utils import compute_metrics

embedding_method = lambda x:  torch.randn(768, generator=torch.Generator().manual_seed(int(hashlib.sha256(x.encode()).hexdigest(), 16) % (2**32)))

class NLI:
    def __init__(self): self.model=type('',(),{'config':type('',(),{'label2id':{'entailment':0,'neutral':1,'contradiction':2}})()})()
    def __call__(self, x, batch_size=64): return [[{'label': l, 'score': 1.0 if i == torch.randint(0, len(self.model.config.label2id), (1,)).item() else 0.0} for i, l in enumerate(self.model.config.label2id)] for _ in x]

nli = NLI()

alpha = 1.0
thr = 0.90
n = 10

g = {
    "original": [{"ans": f"answer_{i}", "logprob": np.log(0.1 + i*0.1)} for i in range(n)],
    "noisy": [{"ans": f"noisy_answer_{i}", "logprob": np.log(0.05 + i*0.05)} for i in range(n)],
    "generated_answer": "main_generated_answer"
}

normal = [d["ans"] for d in g["original"]]
noisy = [d["ans"] for d in g["noisy"]]
logn = [d["logprob"] for d in g["original"]]
logd = [d["logprob"] for d in g["noisy"]]
seq = [g["generated_answer"]] + normal + noisy
gen_dict = {"normal_answers": normal, "noisy_answers": noisy, "normal_log_values": logn, "noisy_log_values": logd}

# ---- embedding-based grouping ----
ids_embd = cluster_terms_by_embedding(seq, embedding_method, threshold=thr)
metrics_embd =  {k + "_embd": v for k, v in compute_metrics(n, ids_embd, normal, noisy, logn, logd, alpha).items()}

# ---- NLI-based grouping ----
ids_nli = cluster_terms_by_nli(seq, nli)
metrics_nli =  {k + "_nli": v for k, v in compute_metrics(n, ids_nli, normal, noisy, logn, logd, alpha).items()}

print("Metrics (embedding-based):", metrics_embd)
print("Metrics (NLI-based):", metrics_nli)