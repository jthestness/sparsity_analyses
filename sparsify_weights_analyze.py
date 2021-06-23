import argparse
import numpy as np
import pdb
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--from_init', action="store_true")
parser.add_argument('--wts_names', action="store_true")
parser.add_argument('--weights', default=None)
parser.add_argument('--debug', action="store_true")
args = parser.parse_args()

if args.from_init:
    wts = np.random.normal(size=(640,640))
else:
    if args.weights is None:
        wts_name = 'decoder/self_attention_3/output_transform/output_transform/kernel'
    else:
        wts_name = args.weights
    model = np.load('outputs/predict_model_size/learning_rate_tests_0/m-57M_b-36_lr-0.00025/model.ckpt-63545.dict.npz')
    if args.wts_names:
        for wts_name in sorted(model.files):
            print(wts_name)
        sys.exit(0)
    wts = model[wts_name]

base_wts_scale = np.linalg.norm(wts)
abs_wts = np.absolute(wts)
sort_wts = np.sort(abs_wts, axis=None)

if args.debug:
    pdb.set_trace()

all_evals = []
all_better_evals = []
row_sparsity = []
column_sparsity = []
for sparsity_level in [0.0, 0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
    nonzero_wts_idx = int(wts.size * sparsity_level)
    min_nonzero_wt = sort_wts[nonzero_wts_idx]

    curr_wts = np.where(abs_wts >= min_nonzero_wt, wts, 0.0)
    curr_nonzero = np.where(abs_wts >= min_nonzero_wt, np.sign(wts), 0.0)

    column_sparsity.append(np.sum(np.abs(curr_nonzero), axis=0))
    row_sparsity.append(np.sum(np.abs(curr_nonzero), axis=-1))

    curr_wts_scale = np.linalg.norm(curr_wts)
    scaled_curr_wts = curr_wts * base_wts_scale / curr_wts_scale
    evals, evecs = np.linalg.eig(scaled_curr_wts)
    evals_magnitudes = np.absolute(evals)
    evals_argsort = np.flip(np.argsort(evals_magnitudes))
    # Sort the eigenvectors with the same order as eigenvalues
    evecs = evecs[evals_argsort]
    evals_magnitudes = evals_magnitudes[evals_argsort]

    all_evals.append(evals_magnitudes)
    if args.debug:
        pdb.set_trace()

for i in range(wts.shape[0]):
    line_str = ''
    for evals in all_evals:
        line_str += f'{evals[i]}\t'
    print(line_str)

print('')

for i in range(wts.shape[0]):
    line_str = ''
    for row_sparse in row_sparsity:
        line_str += f'{row_sparse[i]}\t'
    print(line_str)

print('')

for i in range(wts.shape[0]):
    line_str = ''
    for col_sparse in column_sparsity:
        line_str += f'{col_sparse[i]}\t'
    print(line_str)
