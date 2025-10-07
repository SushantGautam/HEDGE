from algorithms import sentence_semantic_entropy, radflag, vase

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