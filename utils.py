from algorithms import sentence_semantic_entropy, radflag, vase
import hashlib
import torch

dummy_embedding_method = lambda x:  torch.randn(768, generator=torch.Generator().manual_seed(int(hashlib.sha256(x.encode()).hexdigest(), 16) % (2**32)))

class DummyNLI:
    def __init__(self): self.model=type('',(),{'config':type('',(),{'label2id':{'entailment':0,'neutral':1,'contradiction':2}})()})()
    def __call__(self, x, batch_size=64): return [[{'label': l, 'score': 1.0 if i == torch.randint(0, len(self.model.config.label2id), (1,)).item() else 0.0} for i, l in enumerate(self.model.config.label2id)] for _ in x]




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