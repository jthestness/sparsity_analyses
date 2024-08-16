
import argparse
import numpy as np

def softmax(tensor, axis=-1):
    max = tensor.max(axis=axis, keepdims=True)
    exp = np.exp(tensor - max)
    sum = exp.sum(axis=axis, keepdims=True)
    return exp / sum

def layer_norm(tensor, axis=-1):
    mean = tensor.mean(axis=axis, keepdims=True)
    tensor = tensor - mean
    stdev = tensor.std(axis=axis, keepdims=True)
    tensor = tensor / stdev
    return tensor

parser = argparse.ArgumentParser()
parser.add_argument('--num_clusters', type=int, default=8)
parser.add_argument('--data_dimension', type=int, default=3)
parser.add_argument('--noise_level', type=float, default=0.1)
parser.add_argument('--sample_size', type=int, default=256)
parser.add_argument('--use_layer_norm', action="store_true")
parser.add_argument('--num_iters', type=int, default=10)
args = parser.parse_args()

total_samples = args.sample_size * args.num_clusters
sample_swaps = []
for iter in range(args.num_iters):
    clust_vecs = np.random.normal(size=(args.num_clusters, args.data_dimension))

    samples = []
    sample_clust_ids = []

    for clust_id in range(args.num_clusters):
        sample_noise = np.random.normal(size=(args.sample_size, args.data_dimension)) * args.noise_level
        samples.append(clust_vecs[clust_id] + sample_noise)
        sample_clust_ids.append(np.array([clust_id] * args.sample_size))

    samples = np.concatenate(samples)
    sample_clust_ids = np.concatenate(sample_clust_ids)

    if args.use_layer_norm:
        samples = layer_norm(samples)

    # Find closest expert
    samples_clust_align = np.matmul(samples, np.transpose(clust_vecs))
    samples_clust_sm = softmax(samples_clust_align)
    del samples_clust_align
    samples_clust_noised_ids = samples_clust_sm.argmax(axis=-1)
    assert samples_clust_noised_ids.size == total_samples

    sample_swaps.append((sample_clust_ids != samples_clust_noised_ids).sum())

    del clust_vecs
    del sample_clust_ids
    del samples_clust_noised_ids

sample_swaps = np.array(sample_swaps)
std_swaps = sample_swaps.std()
sample_swaps = sample_swaps.mean()

# /= args.num_iters

print(f'Swapped IDs: {args.data_dimension} {args.num_clusters} {args.noise_level} {sample_swaps} {total_samples} {1.0 - sample_swaps / total_samples} {std_swaps}')
