import argparse
import math
import numpy as np

def reconstruct_low_rank(u, s, vh, rank):
    return np.matmul(np.matmul(u[:,:rank], np.diag(s[:rank])), vh[:rank,:])

def create_range(max_val):
    range = []
    i = 1
    while i < max_val:
        range.append(int(i))
        if i == 1:
            i = 2
        elif i % 5 == 0:
            i = i * 6 / 5
        elif i % 3 == 0:
            i = i * 4 / 3
        elif i % 4 == 0:
            i = i * 5 / 4
        else:
            i = i * 3 / 2

        if i - range[-1] > 32:
            i = range[-1] + 32
    range.append(max_val)
    return range

def calc_approx_errors(orig_wts, approx_wts, embed, orig_proj_embed):
    # Direct comparison of original and approximated weights
    recon_dist = np.linalg.norm(approx_wts - orig_wts)
    recon_len_diff = np.absolute(np.linalg.norm(approx_wts) - np.linalg.norm(orig_wts))
    recon_embed = np.matmul(embed, approx_wts)
    recon_embed_dist = np.linalg.norm(recon_embed - orig_proj_embed)
    recon_embed_len_diff = np.absolute(np.linalg.norm(recon_embed) - np.linalg.norm(orig_proj_embed))
    # Correct the magnitude of the approximated weights
    approx_corr_wts = approx_wts * (np.linalg.norm(orig_wts) / np.linalg.norm(approx_wts))
    corr_recon_dist = np.linalg.norm(approx_corr_wts - orig_wts)
    corr_recon_len_diff = np.absolute(np.linalg.norm(approx_corr_wts) - np.linalg.norm(orig_wts))
    corr_recon_embed = np.matmul(embed, approx_corr_wts)
    corr_recon_embed_dist = np.linalg.norm(corr_recon_embed - orig_proj_embed)
    corr_recon_embed_len_diff = np.absolute(np.linalg.norm(corr_recon_embed) - np.linalg.norm(orig_proj_embed))
    return recon_dist, recon_len_diff, recon_embed_dist, recon_embed_len_diff, corr_recon_dist, corr_recon_len_diff, corr_recon_embed_dist, corr_recon_embed_len_diff


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--projection', default='k')
parser.add_argument('--layer', default=0, type=int)
parser.add_argument('--step_id', default=90000, type=int)
parser.add_argument('--interpolate', action='store_true')
args = parser.parse_args()


weights = np.load(f'outputs/running_on_cs1/gpt2_small_msl128_bs144_lr0.00028_gpu-baseline_1/model.ckpt-{args.step_id}.dict.npz')

embed = weights['input_embedding/embedding_weights']
layer_id = ''
if args.layer > 0:
    layer_id = f'_{args.layer}'
proj_wts = weights[f'decoder/self_attention{layer_id}/{args.projection}_projection/{args.projection}_projection/kernel']

# TODO: REMOVE ME!
# Low rank approximate the proj_weights before the rest of this testing!
# u, s, vh = np.linalg.svd(proj_wts)
# proj_wts = reconstruct_low_rank(u, s, vh, 128)

# Pre-process weights for low-rank and pruning
# SVD of projection weights
u, s, vh = np.linalg.svd(proj_wts)
abs_proj_wts = np.absolute(proj_wts)
sort_proj_wts = np.sort(abs_proj_wts, axis=None)

# Process weights we're interested in
# Layer normalize embedding tokens
embsample = np.arange(400) * 125
embed = embed[embsample]

# TODO: REMOVE ME!
# Try setting the embedding to a random initialization
# embed = np.random.normal(size=embed.shape)

# TODO: REMOVE ME!
# Try setting the embedding to the standard basis
# embed = np.eye(proj_wts.shape[0])[:400]

# TODO: REMOVE ME!
# Try setting the embedding to a random orthonormal basis
# from scipy.stats import ortho_group
# embed = ortho_group.rvs(dim=proj_wts.shape[0])[:400]

# TODO: REMOVE ME!
# What happens when the embeddings (activations) are low-rank?
# emb_u, emb_s, emb_vh = np.linalg.svd(embed)
# embed = reconstruct_low_rank(emb_u, emb_s, emb_vh, 50)

embmean = np.mean(embed, axis=-1, keepdims=True)
embstdev = np.std(embed, axis=-1, keepdims=True)
embed = (embed - embmean) / embstdev
proj_embed = np.matmul(embed, proj_wts)


# For each rank 1 to full
#  - Reconstruct weights
#  - Project tokens
#  - Measure error (e.g., cosine sim, length, length of difference)
#  - Project sampled embeddings and measure error
hidden_size = embed.shape[-1]
proj_wts_norm = proj_wts / np.linalg.norm(proj_wts, axis=-1, keepdims=True)
for rank in create_range(hidden_size):
    lr_proj_wts = reconstruct_low_rank(u, s, vh, rank)
    (krecon_dist, krecon_len_diff, krecon_embed_dist, krecon_embed_len_diff,
     kcrecon_dist, kcrecon_len_diff, kcrecon_embed_dist, kcrecon_embed_len_diff) = \
        calc_approx_errors(proj_wts, lr_proj_wts, embed, proj_embed)

    # Compare low-rank with pruning (for context)
    sparsity_level = (1.0 - (2 * hidden_size * rank / hidden_size ** 2))
    sparsity = sparsity_level * 100
    sparsity_level = max(0.0, sparsity_level)
    min_nonzero_k_wt = sort_proj_wts[int(proj_wts.size * sparsity_level)]
    sparse_proj_wts = np.where(abs_proj_wts >= min_nonzero_k_wt, proj_wts, 0.0)

    (sparse_dist, sparse_len_diff, sparse_embed_dist, sparse_embed_len_diff,
     corr_sparse_dist, corr_sparse_len_diff, corr_sparse_embed_dist, corr_sparse_embed_len_diff) = \
        calc_approx_errors(proj_wts, sparse_proj_wts, embed, proj_embed)

    if not args.interpolate:
        print(f'{rank}\t{sparsity:.2f}%\t{krecon_dist:.4f}\t{krecon_len_diff:.4f}\t{krecon_embed_dist:.4f}\t{krecon_embed_len_diff:.4f}\t{kcrecon_embed_dist:.4f}\t{kcrecon_embed_len_diff:.4f}\t{sparse_dist:.4f}\t{sparse_len_diff:.4f}\t{sparse_embed_dist:.4f}\t{sparse_embed_len_diff:.4f}\t{corr_sparse_embed_dist:.4f}\t{corr_sparse_embed_len_diff:.4f}')

    # TODO(joel): We want to analyze how the low-rank and pruned
    # approximations accumulate error as we interpolate from the true weights
    # to the low-rank or pruned values. This would give us some understanding
    # of how narrowly applicable the removed weights are... We expect that
    # error will grow very quickly when interpolating away from true weights
    if args.interpolate:
        for interp in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]:
            krecon_interp = lr_proj_wts * interp + proj_wts * (1.0 - interp)
            (krecon_dist, krecon_len_diff, krecon_embed_dist, krecon_embed_len_diff,
             kcrecon_dist, kcrecon_len_diff, kcrecon_embed_dist, kcrecon_embed_len_diff) = \
                calc_approx_errors(proj_wts, krecon_interp, embed, proj_embed)
            print(f'  LRReconInterp\t{rank}\t{sparsity:.2f}%\t{interp}\t{krecon_dist}\t{krecon_len_diff}\t{krecon_embed_dist}\t{krecon_embed_len_diff}\t{kcrecon_dist}\t{kcrecon_len_diff}\t{kcrecon_embed_dist}\t{kcrecon_embed_len_diff}')

            sparse_interp = sparse_proj_wts * interp + proj_wts * (1.0 - interp)
            (sparse_dist, sparse_len_diff, sparse_embed_dist, sparse_embed_len_diff,
             corr_sparse_dist, corr_sparse_len_diff, corr_sparse_embed_dist, corr_sparse_embed_len_diff) = \
                calc_approx_errors(proj_wts, sparse_interp, embed, proj_embed)
            print(f'  PruneInterp\t{rank}\t{sparsity:.2f}%\t{interp}\t{sparse_dist}\t{sparse_len_diff}\t{sparse_embed_dist}\t{sparse_embed_len_diff}\t{corr_sparse_dist}\t{corr_sparse_len_diff}\t{corr_sparse_embed_dist}\t{corr_sparse_embed_len_diff}')

    if args.debug:
        import pdb
        pdb.set_trace()

if args.debug:
    import pdb
    pdb.set_trace()
