import torch
import numpy as np
from scipy.sparse import coo_matrix as COO
from scipy.sparse.csgraph import connected_components as CC
from sklearn.neighbors import NearestNeighbors as NN
import torch.nn.functional as F
import itertools
from tqdm import tqdm
import gc



def sentence_semantic_entropy(mean_log_liks, semantic_ids, eps=1e-38):
    logls, semids = torch.as_tensor(mean_log_liks, dtype=torch.float32), torch.as_tensor(semantic_ids, dtype=torch.long)
    weight = torch.exp(logls - logls.max())
    _, invix = torch.unique(semids, return_inverse=True, sorted=True)
    groups = torch.zeros(invix.max()+1, dtype=logls.dtype, device=logls.device).scatter_add(0, invix, weight)
    log_group_probs = torch.log(groups.clamp_min(eps)) - torch.log(weight.sum().clamp_min(eps))
    group_probs = log_group_probs.exp()
    entropy = -(group_probs * log_group_probs).sum()
    return entropy, group_probs

def cluster_terms_by_nli(S, nli):
    if not S: return []
    n=len(S);A,B=torch.triu_indices(n,n,1); par=list(range(n))
    S = [ep[:200]for ep in S] # trim to 200 characters to avoid CUDA OOM 
    preds=[max(r,key=lambda d:d["score"])["label"] for r in nli([{"text":S[i],"text_pair":S[j]} for i,j in zip(A,B)] + [{"text":S[j],"text_pair":S[i]} for i,j in zip(A,B)], batch_size=64)]
    P = np.array(preds, dtype=object).reshape(2, -1);     
    eq = torch.from_numpy((P[0] != 'CONTRADICTION') & (P[1] != 'CONTRADICTION') & ~((P[0] == 'NEUTRAL') & (P[1] == 'NEUTRAL')))
    def find(x): 
        while par[x]!=x: par[x]=par[par[x]];x=par[x]
        return x
    for u,v in zip(A[eq].tolist(),B[eq].tolist()):
        ru,rv=find(u),find(v)
        if ru!=rv: par[rv]=ru
    roots=[find(i) for i in range(n)]
    mapping={r:i for i,r in enumerate(dict.fromkeys(roots))}
    return [mapping[r] for r in roots]

##### below is  batched implementation of cluster_terms_by_nli, splitted into get_nli_labels and cluster_from_nli_labels


def get_nli_labels(S_batch, nli, B=256):
    ns    = [len(g) for g in S_batch]
    idx_f = [[(g,i,j) for i in range(n) for j in range(i+1,n)] for g,n in enumerate(ns)]
    idx   = list(itertools.chain.from_iterable(grp + [(g,j,i) for (g,i,j) in grp] for grp in idx_f))
    pairs = [{"text": S_batch[g][i][:200], "text_pair": S_batch[g][j][:200]} for g,i,j in idx]

    # dedup via dict comprehension (preserves order)
    keys  = [(p["text"], p["text_pair"]) for p in pairs]
    uniq  = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    u_pairs = [{"text": k[0], "text_pair": k[1]} for k in uniq]
    map_ix  = [uniq[k] for k in keys]
    print("Total unique pairs for NLI: ", len(u_pairs))
    pu = []
    for b in tqdm((u_pairs[k:k+B] for k in range(0, len(u_pairs), B)),
                total=(len(u_pairs)+B-1)//B, desc="Running NLI", unit="batch"):
        out = nli(b, batch_size=B, truncation=True, top_k=None)  # run one batch
        pu.extend(max(r, key=lambda d: d["score"])["label"] for r in out)  # keep only labels
        del out  # drop logits/probs immediately
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    preds = [pu[i] for i in map_ix] # map backed
    lens   = [n*(n-1) for n in ns]
    splits = list(itertools.accumulate(lens))[:-1] if lens else []
    grouped= (np.split(np.array(preds, dtype=object), splits) if preds else [np.array([], dtype=object) for _ in ns])
    return [{"n": n, "P": (np.empty((2,0), dtype=object) if n == 0 else g.reshape(2, -1))} for n,g in zip(ns, grouped)]

def cluster_from_nli_labels(labels):
    output=[]
    for e in tqdm(labels, desc="Clustering from NLI labels", unit="item"):
        n,P=e["n"],e["P"]
        if n==0: output.append([]); continue
        A,B=torch.triu_indices(n,n,1)
        eq=torch.from_numpy((P[0]!='CONTRADICTION') & (P[1]!='CONTRADICTION') & ~((P[0]=='NEUTRAL') & (P[1]=='NEUTRAL')))
        par=list(range(n))
        def find(x):
            while par[x]!=x: par[x]=par[par[x]]; x=par[x]
            return x
        for u,v in zip(A[eq].tolist(),B[eq].tolist()):
            ru,rv=find(u),find(v)
            if ru!=rv: par[rv]=ru
        roots=[find(i) for i in range(n)]
        mapping={r:i for i,r in enumerate(dict.fromkeys(roots))}
        output.append([mapping[r] for r in roots])
    return output

def cluster_terms_by_embedding(S, embed_method, threshold=None, topk=None, metric="cosine"):
    E = F.normalize(torch.stack([embed_method(t) for t in S]).float(), p=2, dim=-1).numpy()
    D, N = NN(n_neighbors=topk or len(E), metric=metric).fit(E).kneighbors(E)
    r, c = np.where((1 - D >= threshold) & (N != np.arange(len(E))[:, None])); c = N[r, c]
    G = COO((np.ones_like(r), (r, c)), shape=(len(E),) * 2); G = G + G.T
    return CC(G, directed=False)[1].tolist()

def radflag(semantic_ids, n):
    semantic_subset = semantic_ids[1:1+n]
    return semantic_subset.count(0) / n

def vase(n, semantic_ids, SeDist, SeDist_noisy, alpha):
    all_ids, dist_vec = torch.as_tensor(semantic_ids), torch.zeros(int(max(semantic_ids))+1)
    dist_clean, dist_noisy = dist_vec.scatter(0, all_ids[1:n+1].unique(), SeDist), dist_vec.scatter(0, all_ids[n+1:2*n+1].unique(), SeDist_noisy)
    ViSeDist = torch.softmax(dist_clean+alpha*(dist_clean-dist_noisy),0)
    return -(ViSeDist*(ViSeDist+1e-10).log()).sum().item()


if __name__ == "__main__":
    print("this is just a utils file, dont run it directly")