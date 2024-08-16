
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
parser.add_argument('--num_experts', type=int, default=4)
parser.add_argument('--sample_size', type=int, default=256)
parser.add_argument('--use_layer_norm', action="store_true")
args = parser.parse_args()

clust_vecs = np.random.normal(size=(args.num_clusters, args.data_dimension))
expert_vecs = np.random.normal(size=(args.num_experts, args.data_dimension))

sample_clust_ids = np.random.choice(np.arange(args.num_clusters), size=(args.sample_size))
sample_noise = np.random.normal(size=(args.sample_size, args.data_dimension)) * args.noise_level
samples = clust_vecs[sample_clust_ids] + sample_noise
if args.use_layer_norm:
    samples = layer_norm(samples)

# Find closest expert
expert_align = np.matmul(samples, np.transpose(expert_vecs))
expert_sm = softmax(expert_align)
expert_select = expert_sm.argmax(axis=-1)

mean_stdev = 0.0
mean_lengths = 0.0
num_samples_per_expert = []
for expert_idx in range(args.num_experts):
    samples_to_experts = np.squeeze(np.argwhere(expert_select == expert_idx))
    if samples_to_experts.size == 0:
        continue
    num_samples_per_expert.append(samples_to_experts.size)
    gather_samples = samples[samples_to_experts]
    mean_stdev += gather_samples.std(axis=0).mean() / args.num_experts
    mean_vec = gather_samples.mean(axis=0)
    mean_lengths += np.linalg.norm(mean_vec) / args.num_experts
    print(gather_samples.shape[0], gather_samples.std(axis=0), gather_samples.std(axis=0).mean(), np.linalg.norm(mean_vec))

print(f'Experts stdev: {args.data_dimension} {args.num_clusters} {args.noise_level} {args.num_experts} {args.sample_size} {mean_stdev} {mean_lengths}')
print(f'{np.array(num_samples_per_expert).mean()} {np.array(num_samples_per_expert).std()} {np.array(num_samples_per_expert).std() / np.array(num_samples_per_expert).mean()}')
