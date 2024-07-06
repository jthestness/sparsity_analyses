
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dimension', type=int, default=16)
parser.add_argument('--full_grad', action="store_true")
parser.add_argument('--num_clusters', type=int, default=None)
args = parser.parse_args()

data_dimension = args.data_dimension
if not args.full_grad:
    max_adjust = 0.0
    max_div = 1.0
else:
    max_adjust = 1.0
    max_div = 2.0
eye_correct = (1.0 + max_adjust) / max_div

iter_div = min(data_dimension, 4)
upper_mul = 4
num_clusters_gen = range(data_dimension // iter_div, data_dimension * upper_mul + 1, data_dimension // iter_div)
if args.num_clusters is not None:
    num_clusters_gen = [args.num_clusters]

for num_clusters in num_clusters_gen:
    means = []
    maxes = []
    clust_vecs = np.random.normal(size=(num_clusters, data_dimension)) / np.sqrt(data_dimension)
    clust_vecs = clust_vecs / np.linalg.norm(clust_vecs, axis=-1)[:,None]
    min_max = (1.0 + max_adjust) / max_div
    count = 0
    learning_rate = 0.01
    for num_iters in range(200000):
        # breakpoint()
        prod = (np.matmul(clust_vecs, np.transpose(clust_vecs)) + max_adjust) / max_div
        prod -= (np.eye(num_clusters) * eye_correct)
        curr_max = prod.max()
        # print(f'{curr_max}')
        if min_max > curr_max:
            min_max = curr_max
            count = 0
        else:
            if num_iters > 1000:
                count += 1
            # if learning_rate > 0.00001 and count >= 200:
            #     learning_rate /= 1.4142
            #     count = 0
            if count >= 10000:
                break

        diff = clust_vecs[:,None,:] - clust_vecs[None,:,:]
        grad = prod[:,:,None] * diff
        new_clust_vecs = clust_vecs + learning_rate * grad.sum(axis=1) / num_clusters
        new_clust_vecs = new_clust_vecs / np.linalg.norm(new_clust_vecs, axis=-1)[:,None]
        clust_vecs = new_clust_vecs


    # print(clust_vecs)
    prod = np.matmul(clust_vecs, np.transpose(clust_vecs))
    prod -= np.eye(num_clusters) * 2.0
    # breakpoint()
    print(num_clusters, prod.max(), np.degrees(np.arccos(prod.max())), num_iters)
