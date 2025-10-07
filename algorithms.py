import torch
import numpy as np
from scipy.sparse import coo_matrix as COO
from scipy.sparse.csgraph import connected_components as CC
from sklearn.neighbors import NearestNeighbors as NN
import torch.nn.functional as F

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
    n=len(S);A,B=torch.triu_indices(n,n,1)
    preds=[max(r,key=lambda d:d["score"])["label"] for r in nli([{"text":S[i],"text_pair":S[j]} for i,j in zip(A,B)] + [{"text":S[j],"text_pair":S[i]} for i,j in zip(A,B)], batch_size=64)]
    labs=torch.tensor([nli.model.config.label2id[p] for p in preds]).reshape(2,-1)
    eq=(labs[0]!=0)&(labs[1]!=0)&~((labs[0]==1)&(labs[1]==1));par=list(range(n))
    def find(x): 
        while par[x]!=x: par[x]=par[par[x]];x=par[x]
        return x
    for u,v in zip(A[eq].tolist(),B[eq].tolist()):
        ru,rv=find(u),find(v)
        if ru!=rv: par[rv]=ru
    roots=[find(i) for i in range(n)]
    mapping={r:i for i,r in enumerate(dict.fromkeys(roots))}
    return [mapping[r] for r in roots]

def cluster_terms_by_embedding(S, embed_method, threshold=None, topk=None, metric="cosine"):
    E = F.normalize(torch.stack([embed_method(t.strip().lower()) for t in S]).float(), p=2, dim=-1).numpy()
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