

import argparse
import numpy as np
from math import comb

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=4)
parser.add_argument('--sparsity_level', type=float, default=0.75)
parser.add_argument('--num_samples', type=int, default=1000)
args = parser.parse_args()


def mask_ids_to_mask(mask_ids, mask_shape):
    mask = np.zeros(mask_shape, dtype=np.int)
    for mask_id in mask_ids:
        mask_id_x = int(mask_id / mask_shape[0])
        mask_id_y = mask_id % mask_shape[0]
        mask[mask_id_x, mask_id_y] = 1
    return mask


total_params = args.hidden_size ** 2
num_nonzeros = int(total_params * (1.0 - args.sparsity_level))
true_sparsity = 1.0 - num_nonzeros / total_params
total_masks = comb(total_params, num_nonzeros)

print(f'Total parameters: {total_params}')
print(f'Non-zero params: {num_nonzeros}')
print(f'Sparsity level: {true_sparsity} (requested: {args.sparsity_level})')
print(f'Number of sparsity masks with this sparsity: {total_masks}')

weights = np.random.normal(size=(args.hidden_size, args.hidden_size))

breakpoint()

weight_ids = np.arange(total_params)

largest_sv = np.zeros(args.num_samples)
sv_gap = np.zeros(args.num_samples)
smallest_sv = np.zeros(args.num_samples)

for i in range(args.num_samples):
    mask_ids = np.random.choice(weight_ids, size=num_nonzeros, replace=False)
    mask = mask_ids_to_mask(mask_ids, weights.shape)
    # print(mask)
    masked_weights = weights * mask
    s = np.linalg.svd(masked_weights, compute_uv=False)
    largest_sv[i] = s[0]
    sv_gap[i] = s[0] - s[1]
    smallest_sv[i] = np.where(s != 0, s, 10000000).min()
    # print(s)

print(f'\nSpectral gap histo')
sv_gap_histo = np.histogram(sv_gap)
for i in range(sv_gap_histo[0].shape[0]):
    print(f'{sv_gap_histo[1][i]}\t{sv_gap_histo[0][i]}')

print(f'\nSmallest SV histo')
smallest_sv_histo = np.histogram(smallest_sv)
for i in range(smallest_sv_histo[0].shape[0]):
    print(f'{smallest_sv_histo[1][i]}\t{smallest_sv_histo[0][i]}')

print(f'\nLargest SV histo')
largest_sv_histo = np.histogram(largest_sv)
for i in range(largest_sv_histo[0].shape[0]):
    print(f'{largest_sv_histo[1][i]}\t{largest_sv_histo[0][i]}')

breakpoint()
