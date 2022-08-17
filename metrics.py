import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def vstrans(d1, d2):
    # Get ranks of counts in diagonal
    ranks_1 = np.argsort(d1) + 1
    ranks_2 = np.argsort(d2) + 1
    # Scale ranks betweeen 0 and 1
    nranks_1 = ranks_1 / max(ranks_1)
    nranks_2 = ranks_2 / max(ranks_2)
    nk = len(ranks_1)
    r2k = np.sqrt(np.var(nranks_1 / nk) * np.var(nranks_2 / nk))
    #print("r2k", r2k)
    return r2k

def get_scc(mat1, mat2):
    N = mat1.shape[0]
    corr_diag = np.zeros(len(range(N)))
    weight_diag = corr_diag.copy()
    for d in range(N):
        d1 = mat1.diagonal(d)
        d2 = mat2.diagonal(d)
        
        if (d == 0) or (d == (N - 1)):
            corr_diag[d] = 0
        else:
        # Compute raw pearson coeff for this diag
            corr = np.corrcoef(d1, d2)[0,1]
            if np.isnan(corr):
                if np.std(d1) < 1e-5 and np.std(d2) < 1e-5:
                    corr = 1.0
                else: 
                    corr = 0.0
            corr_diag[d] = corr
            
        # Compute weight for this diag
        r2k = vstrans(d1, d2)
        weight_diag[d] = len(d1) * r2k
    # Normalize weights
    #print("weight diag before division: ", weight_diag)
    weight_diag /= sum(weight_diag)
    #print("weight diag after division: ", weight_diag)

    # Weighted sum of coefficients to get SCCs
    scc = np.sum(corr_diag * weight_diag)
    #print("corr diag: ", corr_diag)
    #print("weight diag: ", weight_diag)
    #print("scc: ", scc)
    return scc

def get_scores(preds, targets):

    preds = preds.cpu()
    targets = targets.cpu()
    
    SCCs = []
    for mat1, mat2 in zip(preds, targets):
        SCC = get_scc(mat1.squeeze(0).numpy(), mat2.squeeze(0).numpy())
        #print("scc: ", SCC)
        SCCs.append(SCC)

    targets = np.reshape(targets, (-1,))
    preds = np.reshape(preds, (-1,))
    
    scores = {
        "mae": mean_absolute_error(targets, preds),
        "mse": mean_squared_error(targets, preds),
        "pearson": pearsonr(targets, preds)[0],
        "spearman": spearmanr(targets, preds)[0],
        "scc":np.mean(SCCs),
    }
    return scores


#if __name__ == '__main__':

     #mat1 = torch.rand((4, 1, 16, 16))
     #mat2 = torch.ones((4, 1, 16, 16))

     #print(get_scores(mat1, mat2))