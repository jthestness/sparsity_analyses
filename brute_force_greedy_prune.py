import argparse
import math
import numpy as np
from tqdm import tqdm

def calc_approx_errors(orig_wts, approx_wts, embed, orig_proj_embed):
    # Direct comparison of original and approximated weights
    # recon_dist = np.linalg.norm(approx_wts - orig_wts)
    # recon_len_diff = np.absolute(np.linalg.norm(approx_wts) - np.linalg.norm(orig_wts))
    recon_embed = np.matmul(embed, approx_wts)
    recon_embed_dist = np.linalg.norm(recon_embed - orig_proj_embed)
    # recon_embed_len_diff = np.absolute(np.linalg.norm(recon_embed) - np.linalg.norm(orig_proj_embed))
    # Correct the magnitude of the approximated weights
    approx_corr_wts = approx_wts * (np.linalg.norm(orig_wts) / np.linalg.norm(approx_wts))
    # corr_recon_dist = np.linalg.norm(approx_corr_wts - orig_wts)
    # corr_recon_len_diff = np.absolute(np.linalg.norm(approx_corr_wts) - np.linalg.norm(orig_wts))
    corr_recon_embed = np.matmul(embed, approx_corr_wts)
    corr_recon_embed_dist = np.linalg.norm(corr_recon_embed - orig_proj_embed)
    # corr_recon_embed_len_diff = np.absolute(np.linalg.norm(corr_recon_embed) - np.linalg.norm(orig_proj_embed))
    return recon_embed_dist, corr_recon_embed_dist


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--projection', default='k')
parser.add_argument('--layer', default=0, type=int)
parser.add_argument('--step_id', default=90000, type=int)
args = parser.parse_args()


weights = np.load(f'outputs/running_on_cs1/gpt2_small_msl128_bs144_lr0.00028_gpu-baseline_1/model.ckpt-{args.step_id}.dict.npz')

embed = weights['input_embedding/embedding_weights']
layer_id = ''
if args.layer > 0:
    layer_id = f'_{args.layer}'
proj_wts = weights[f'decoder/self_attention{layer_id}/{args.projection}_projection/{args.projection}_projection/kernel']
abs_proj_wts = np.absolute(proj_wts)
argsort_proj_wts = np.argsort(abs_proj_wts, axis=None)

# Process weights we're interested in
# Layer normalize embedding tokens
embsample = np.arange(400) * 125
embed = embed[embsample]
embmean = np.mean(embed, axis=-1, keepdims=True)
embstdev = np.std(embed, axis=-1, keepdims=True)
embed = (embed - embmean) / embstdev
proj_embed = np.matmul(embed, proj_wts)

# TODO: There might be an even faster way to do this by subtracting the
# modified column from the partial_result matrix below rather than
# multiplying the modified weights column by the whole embedding matrix
# again...
# TODO: Make the algorithm dynamic: If it finds a new best_error as it
# is walking through the small weights, extend the set of weights to
# consider past the current wt by, say, 100 more weights

curr_wts = proj_wts.copy()
# TODO: curr_corr_wts = proj_wts.copy()
top_k = 200
for iter in range(proj_wts.size):
    partial_result = np.matmul(embed, curr_wts)
    emb_proj_errors = np.zeros((top_k))
    for argmin_position in range(min(top_k, argsort_proj_wts.size)):
        argmin_idx = argsort_proj_wts[argmin_position]
        col_id = argmin_idx % proj_wts.shape[1]
        row_id = argmin_idx // proj_wts.shape[1]

        if curr_wts[row_id, col_id] != 0.0:
            curr_col = curr_wts[:,col_id].copy()
            curr_col[row_id] = 0.0
            recon_embed = partial_result.copy()
            recon_embed[:,col_id] = np.matmul(embed, curr_col)
            emb_proj_error = np.linalg.norm(recon_embed - proj_embed)
            emb_proj_errors[argmin_position] = emb_proj_error
    min_wt = curr_wts[argsort_proj_wts[0] // proj_wts.shape[1], argsort_proj_wts[0] % proj_wts.shape[1]]
    min_wt_error = emb_proj_errors[0]
    best_error = emb_proj_errors.min()
    best_error_argmin_position = emb_proj_errors.argmin()
    argmin_idx = argsort_proj_wts[best_error_argmin_position]
    col_id = argmin_idx % proj_wts.shape[1]
    row_id = argmin_idx // proj_wts.shape[1]
    best_error_wt = curr_wts[row_id, col_id]
    curr_wts[row_id, col_id] = 0.0
    argsort_proj_wts = np.delete(argsort_proj_wts, best_error_argmin_position)
    sparsity = float(iter) / proj_wts.size
    print(f'{sparsity}\t{best_error_argmin_position}\t{argmin_idx}\t{best_error}\t{min_wt_error}\t{min_wt}\t{best_error_wt}')
#    import pdb
#    pdb.set_trace()

if args.debug:
    import pdb
    pdb.set_trace()
