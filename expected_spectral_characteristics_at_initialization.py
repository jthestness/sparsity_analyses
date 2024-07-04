

import numpy as np

def mask_ids_to_mask(mask_ids, mask_shape):
    mask = np.zeros(mask_shape, dtype=np.int)
    for mask_id in mask_ids:
        mask_id_x = int(mask_id / mask_shape[0])
        mask_id_y = mask_id % mask_shape[0]
        mask[mask_id_x, mask_id_y] = 1
    return mask

def magnitude_prune_mask(num_nonzeros, weights):
    abs_wt = np.absolute(weights)
    abs_sorted = np.sort(abs_wt.flatten())
    if num_nonzeros == abs_sorted.size:
       min_magnitude = 0.0
    elif num_nonzeros == 0:
       min_magnitude = abs_sorted[-1] + 1
    else:
       min_magnitude = abs_sorted[min(abs_sorted.size - num_nonzeros, abs_sorted.size-1)]
    mask = abs_wt >= min_magnitude
    return mask

num_samples = 200
num_density_levels = 16
density_levels = np.power(2.0, -np.arange(num_density_levels) / 2.0)
sparsify_techniques = ['random', 'magnitude']

for i in [10, 11]: # range(10):
    first_sv = np.zeros((len(sparsify_techniques), num_density_levels, num_samples))
    second_sv = np.zeros((len(sparsify_techniques), num_density_levels, num_samples))
    nuclear_norms = np.zeros((len(sparsify_techniques), num_density_levels, num_samples))
    spectral_gaps = np.zeros((len(sparsify_techniques), num_density_levels, num_samples))
    for dense_idx, density in enumerate(density_levels):

        for samp in range(num_samples):
            matrix_shape = (2**(i+1), 2**(i+1))
            matrix = np.random.normal(size=matrix_shape)

            for spars_idx, sparsify_technique in enumerate(sparsify_techniques):
                weight_ids = np.arange(matrix.size)
                num_nonzeros = int(matrix.size * density)
                if sparsify_technique == 'random':
                    if num_nonzeros == matrix.size:
                        mask = np.ones_like(matrix)
                    else:
                        mask_ids = np.random.choice(weight_ids, size=num_nonzeros, replace=False)
                        mask = mask_ids_to_mask(mask_ids, matrix.shape)
                elif sparsify_technique == 'magnitude':
                    mask = magnitude_prune_mask(num_nonzeros, matrix)
                else:
                    raise NotImplementedError(f'Sparsify technique: {sparsify_technique}')

                if num_nonzeros != mask.sum():
                    breakpoint()
                sparsified_matrix = matrix * mask

                s_arr = np.linalg.svd(sparsified_matrix, compute_uv=False)
                nuclear_norm = np.linalg.norm(sparsified_matrix, ord='nuc')
                # print(i+1, s_arr[0], s_arr[1], nuclear_norm, s_arr[0] / s_arr[-1], (s_arr[0] - s_arr[1]) / nuclear_norm)
                first_sv[spars_idx][dense_idx][samp] = s_arr[0]
                second_sv[spars_idx][dense_idx][samp] = s_arr[1]
                nuclear_norms[spars_idx][dense_idx][samp] = nuclear_norm
                spectral_gaps[spars_idx][dense_idx][samp] = (s_arr[0] - s_arr[1]) / nuclear_norm

        # TODO: Print maxes/mins of appropriate stats (e.g., we want to know how much larger the max first_sv is than than mean

        print(2**(i+1), 1.0 - density, sparsify_techniques[0], first_sv[0][dense_idx].mean(), first_sv[0][dense_idx].std(), second_sv[0][dense_idx].mean(), second_sv[0][dense_idx].std(), nuclear_norms[0][dense_idx].mean(), nuclear_norms[0][dense_idx].std(), spectral_gaps[0][dense_idx].mean(), spectral_gaps[0][dense_idx].std())
        print(2**(i+1), 1.0 - density, sparsify_techniques[1], first_sv[1][dense_idx].mean(), first_sv[1][dense_idx].std(), second_sv[1][dense_idx].mean(), second_sv[1][dense_idx].std(), nuclear_norms[1][dense_idx].mean(), nuclear_norms[1][dense_idx].std(), spectral_gaps[1][dense_idx].mean(), spectral_gaps[1][dense_idx].std())
