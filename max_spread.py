
import argparse
import numpy as np

# A script that can generate num_clusters vectors of data_dimension
# and they are spread out in the space as much as possible. Calculates
# the minimum angle between any two vectors.

# Some things we've learned:
# - For data_dimension=2, the min angle between vectors is 360/num_clusters degrees
# - For data_dimension>2, the min angle between vectors for num_clusters=data_dimension+1 is angle = arccos(-1/data_dimension)
#     - This is because num_clusters=data_dimension vectors can spread out in data_dimension-1 space (i.e., recursive application of this)
# - For data_dimension>2, min angle between num_clusters vectors num_clusters in [data_dimension+2, 2*data_dimension] is 90 degrees
#     - This is because of recursive application of this property with orthogonal subspaces

parser = argparse.ArgumentParser()
parser.add_argument('--data_dimension', type=int, default=16)
parser.add_argument('--full_grad', action="store_true")
parser.add_argument('--num_clusters', type=int, default=None)
parser.add_argument('--num_trials', type=int, default=1)
parser.add_argument('--use_softmax', action="store_true")
args = parser.parse_args()


def softmax(tensor, axis=-1):
    max = tensor.max(axis=axis, keepdims=True)
    tens_max = tensor - max
    exp = np.exp(tens_max)
    sum = exp.sum(axis=axis, keepdims=True)
    return exp / sum

data_dimension = args.data_dimension
num_trials = args.num_trials
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
    for trial in range(num_trials):
        means = []
        maxes = []
        clust_vecs = np.random.normal(size=(num_clusters, data_dimension)) / np.sqrt(data_dimension)
        clust_vecs = clust_vecs / np.linalg.norm(clust_vecs, axis=-1)[:,None]
        min_max = (1.0 + max_adjust) / max_div
        count = 0
        learning_rate = 0.02
        for num_iters in range(300000):
            # breakpoint()
            prod = (np.matmul(clust_vecs, np.transpose(clust_vecs)) + max_adjust) / max_div
            prod -= (np.eye(num_clusters) * eye_correct)
            if args.use_softmax:
                prod = softmax(prod, axis=-1)
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
        print(data_dimension, num_clusters, prod.max(), np.degrees(np.arccos(prod.max())), num_iters)
