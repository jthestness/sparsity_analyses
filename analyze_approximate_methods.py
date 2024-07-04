import argparse
import math
import numpy as np
import os

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

num_bins = 300
bins_range = 50
scale = float(num_bins) / float(bins_range)
bins = (np.arange(num_bins + 1) - (num_bins/2.0)) / scale
centers = (bins[:-1] + bins[1:]) / 2.0

def calc_kl_divergence(histo_a, histo_b, epsilon=1e-15):
    assert np.all(histo_a[1] == histo_b[1])
    log_ratio = np.log(histo_a[0] / (histo_b[0] + epsilon))
    log_ratio = np.where(np.isfinite(log_ratio), log_ratio, 0.0)
    prob_mass = histo_a[0] * (histo_a[1][1:] - histo_a[1][:-1])
    kl_div = prob_mass * np.absolute(log_ratio)
    return kl_div.sum()

def calc_approx_errors(orig_wts, approx_wts, embed, orig_proj_embed, rescale_weights=False):
    # Whether to rescale weights
    if rescale_weights:
        approx_wts = approx_wts * (np.linalg.norm(orig_wts) / np.linalg.norm(approx_wts))
    # Direct comparison of original and approximated weights
    recon_dist = np.linalg.norm(approx_wts - orig_wts)
    recon_len_diff = np.absolute(np.linalg.norm(approx_wts) - np.linalg.norm(orig_wts))
    recon_embed = np.matmul(embed, approx_wts)
    recon_embed_dist = np.linalg.norm(recon_embed - orig_proj_embed)
    recon_embed_len_diff = np.absolute(np.linalg.norm(recon_embed) - np.linalg.norm(orig_proj_embed))
    # TODO(joel): Add 1-norm, inf-norm and return
    recon_embed_histo = np.histogram(recon_embed, density=True, bins=bins)
    kl_div = calc_kl_divergence(proj_embed_histo, recon_embed_histo)
    # TODO(joel): Add norm of difference between
    # NOTE: My intuition is screaming that we're pruning weights in a silly
    # way for transformer models or at least for self-attention mechanisms...
    # Pruning single weights perturbs token/embed vectors all in a
    # systematically biased direction, so relationships between vectors
    # collapse!
    #   - np.matmul(orig_proj_embed, np.transpose(orig_proj_embed))
    #   - np.matmul(np.matmul(approx_wts, embed), np.transpose(orig_proj_embed))
    #   This is a measure of how much the dot products between vectors change (e.g., like attention)
    return recon_dist, recon_len_diff, recon_embed_dist, recon_embed_len_diff, kl_div


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--projection', default='k')
parser.add_argument('--layer', default=0, type=int)
parser.add_argument('--step_id', default=90000, type=int)
parser.add_argument('--interpolate', action='store_true')
parser.add_argument('--input_apply_ln_weights', action='store_true')
parser.add_argument('--input_random', action='store_true')
parser.add_argument('--input_standard_basis', action='store_true')
parser.add_argument('--input_orthonormal_basis', action='store_true')
parser.add_argument('--input_reduce_rank', default=0, type=int)
parser.add_argument('--sample_input', default=400, type=int)
parser.add_argument('--weights_reduce_rank', default=0, type=int)
parser.add_argument('--weights_rescale', action='store_true')
args = parser.parse_args()

outdir = 'outputs/running_on_cs1/gpu/gpt2_small_msl128_bs144_lr0.00028_0'
wts_file = os.path.join(outdir, f'model.ckpt-{args.step_id}.dict.npz')
weights = np.load(wts_file)

embed = weights['input_embedding/embedding_weights']
layer_id = ''
if args.layer > 0:
    layer_id = f'_{args.layer}'
proj_wts = weights[f'decoder/self_attention{layer_id}/{args.projection}_projection/{args.projection}_projection/kernel']
ln_gamma = weights['decoder/layer_normalization_layer/layer_normalization_layer/gamma']
ln_beta = weights['decoder/layer_normalization_layer/layer_normalization_layer/beta']

# Low rank approximate the proj_weights before the rest of this testing
if args.weights_reduce_rank > 0:
    u, s, vh = np.linalg.svd(proj_wts)
    proj_wts = reconstruct_low_rank(u, s, vh, args.weights_reduce_rank)

# Pre-process weights for low-rank and pruning
# SVD of projection weights
u, s, vh = np.linalg.svd(proj_wts)
abs_proj_wts = np.absolute(proj_wts)
sort_proj_wts = np.sort(abs_proj_wts, axis=None)

# Process weights we're interested in
# Layer normalize embedding tokens
if args.sample_input > 0:
    offset = embed.shape[0] // args.sample_input
    embsample = np.arange(args.sample_input) * offset
    embed = embed[embsample]

# Try setting the embedding to a random initialization
if args.input_random:
    embed = np.random.normal(size=embed.shape)

# Try setting the embedding to the standard basis
if args.input_standard_basis:
    embed = np.eye(proj_wts.shape[0])[:embed.shape[-1]]

# Try setting the embedding to a random orthonormal basis
if args.input_orthonormal_basis:
    from scipy.stats import ortho_group
    embed = ortho_group.rvs(dim=proj_wts.shape[0])[:embed.shape[-1]]

# What happens when the embeddings (activations) are low-rank?
if args.input_reduce_rank > 0:
    emb_u, emb_s, emb_vh = np.linalg.svd(embed)
    embed = reconstruct_low_rank(emb_u, emb_s, emb_vh, args.input_reduce_rank)

embmean = np.mean(embed, axis=-1, keepdims=True)
embstdev = np.std(embed, axis=-1, keepdims=True)
embed = (embed - embmean) / embstdev
if args.input_apply_ln_weights:
    embed *= ln_gamma
    embed += ln_beta
embed_histo = np.histogram(embed, density=True, bins=bins)
proj_embed = np.matmul(embed, proj_wts)
proj_embed_histo = np.histogram(proj_embed, density=True, bins=bins)
in_out_kl_div = calc_kl_divergence(proj_embed_histo, embed_histo)
print(f'Input-to-output distribution KL divergence: {in_out_kl_div}')

import pdb
pdb.set_trace()

# For each rank 1 to full
#  - Reconstruct weights
#  - Project tokens
#  - Measure error (e.g., cosine sim, length, length of difference)
#  - Project sampled embeddings and measure error
hidden_size = embed.shape[-1]
proj_wts_norm = proj_wts / np.linalg.norm(proj_wts, axis=-1, keepdims=True)
lowrank_embed_histos = []
sparse_embed_histos = []
for rank in create_range(hidden_size):
    lr_proj_wts = reconstruct_low_rank(u, s, vh, rank)
    (krecon_dist, krecon_len_diff, krecon_embed_dist, krecon_embed_len_diff,
     krecon_kl_div) = \
        calc_approx_errors(proj_wts, lr_proj_wts, embed, proj_embed, args.weights_rescale)
    lowrank_embed = np.matmul(embed, lr_proj_wts)
    lowrank_embed_histos.append(np.histogram(lowrank_embed, density=True, bins=bins))

    # Compare low-rank with pruning (for context)
    sparsity_level = (1.0 - (2 * hidden_size * rank / hidden_size ** 2))
    sparsity = sparsity_level * 100
    sparsity_level = max(0.0, sparsity_level)
    min_nonzero_k_wt = sort_proj_wts[int(proj_wts.size * sparsity_level)]
    sparse_proj_wts = np.where(abs_proj_wts >= min_nonzero_k_wt, proj_wts, 0.0)

    (sparse_dist, sparse_len_diff, sparse_embed_dist, sparse_embed_len_diff,
     sparse_kl_div) = \
        calc_approx_errors(proj_wts, sparse_proj_wts, embed, proj_embed, args.weights_rescale)
    sparse_embed = np.matmul(embed, sparse_proj_wts)
    sparse_embed_histos.append(np.histogram(sparse_embed, density=True, bins=bins))

    if not args.interpolate:
        print(f'{rank}\t{sparsity:.2f}%\t{krecon_dist:.4f}\t{krecon_len_diff:.4f}\t{krecon_embed_dist:.4f}\t{krecon_embed_len_diff:.4f}\t{sparse_dist:.4f}\t{sparse_len_diff:.4f}\t{sparse_embed_dist:.4f}\t{sparse_embed_len_diff:.4f}\t\t\t{krecon_kl_div:.6f}\t{sparse_kl_div:.6f}')

    # TODO(joel): We want to analyze how the low-rank and pruned
    # approximations accumulate error as we interpolate from the true weights
    # to the low-rank or pruned values. This would give us some understanding
    # of how narrowly applicable the removed weights are... We expect that
    # error will grow very quickly when interpolating away from true weights
    if args.interpolate:
        for interp in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]:
            krecon_interp = lr_proj_wts * interp + proj_wts * (1.0 - interp)
            (krecon_dist, krecon_len_diff, krecon_embed_dist, krecon_embed_len_diff,
             krecon_kl_div) = \
                calc_approx_errors(proj_wts, krecon_interp, embed, proj_embed, args.weights_rescale)
            print(f'  LRReconInterp\t{rank}\t{sparsity:.2f}%\t{interp}\t{krecon_dist:.4f}\t{krecon_len_diff:.4f}\t{krecon_embed_dist:.4f}\t{krecon_embed_len_diff:.4f}\t{krecon_kl_div:.4f}')

            sparse_interp = sparse_proj_wts * interp + proj_wts * (1.0 - interp)
            (sparse_dist, sparse_len_diff, sparse_embed_dist, sparse_embed_len_diff,
             sparse_kl_div) = \
                calc_approx_errors(proj_wts, sparse_interp, embed, proj_embed, args.weights_rescale)
            print(f'  PruneInterp\t{rank}\t{sparsity:.2f}%\t{interp}\t{sparse_dist:.4f}\t{sparse_len_diff:.4f}\t{sparse_embed_dist:.4f}\t{sparse_embed_len_diff:.4f}\t{sparse_kl_div:.4f}')

            sparse_interp = sparse_proj_wts * interp + np.eye(sparse_proj_wts.shape[0]) * (1.0 - interp)
            (sparse_dist, sparse_len_diff, sparse_embed_dist, sparse_embed_len_diff,
             sparse_kl_div) = \
                calc_approx_errors(proj_wts, sparse_interp, embed, proj_embed, args.weights_rescale)
            print(f'  PruneInterpIdent\t{rank}\t{sparsity:.2f}%\t{interp}\t{sparse_dist:.4f}\t{sparse_len_diff:.4f}\t{sparse_embed_dist:.4f}\t{sparse_embed_len_diff:.4f}\t{sparse_kl_div:.4f}')

    if args.debug:
        import pdb
        pdb.set_trace()

import sys
sys.exit(0)

print('\n')
for i in range(lowrank_embed_histos[0][1].size - 1):
    str_to_print = f'{i}'
    for lr_histo in lowrank_embed_histos:
        str_to_print += f'\t{lr_histo[0][i]:.4f}'
    print(str_to_print)

print('\n')
for i in range(sparse_embed_histos[0][1].size - 1):
    str_to_print = f'{i}'
    for se_histo in sparse_embed_histos:
        str_to_print += f'\t{se_histo[0][i]:.4f}'
    print(str_to_print)

if args.debug:
    import pdb
    pdb.set_trace()
