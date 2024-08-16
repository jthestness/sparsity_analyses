
import argparse
import numpy as np
import torch


class MLPModel(torch.nn.Module):
    def __init__(self, num_clusters, hidden_size):
        super(MLPModel, self).__init__()
        self.hidden_layer = torch.nn.Linear(hidden_size, num_clusters)

    def forward(self, inputs):
        return self.hidden_layer(inputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--noise_coefficient', type=float, default=0.1)
    args = parser.parse_args()

    num_clusters = args.num_clusters
    hidden_size = args.hidden_size
    noise_coefficient = args.noise_coefficient

    cluster_vectors = np.random.normal(size=(num_clusters, hidden_size)) / np.sqrt(hidden_size)

    my_model = MLPModel(hidden_size=hidden_size, num_clusters=num_clusters)

    num_iters = 100
    batch_size = 64
    for i in range(num_iters):
        # Generate randomized data vectors
        cluster_ids = np.random.choice(np.arange(num_clusters), size=(batch_size))
        batch_vectors = cluster_vectors[cluster_ids]
        noise_vectors = np.random.normal(size=(batch_size, hidden_size)) * noise_coefficient
        batch_vectors += noise_vectors
        batch_vectors = torch.tensor(batch_vectors, dtype=torch.float)

        my_outs = my_model(batch_vectors)
        my_outs = torch.softmax(my_outs, axis=-1)
        # my_clusters = torch.nn.CrossEntropyLoss(my_outs, cluster_ids)
        my_preds = torch.zeros(size=(batch_size,))
        for j in range(batch_size):
            my_preds[j] = my_outs[j][cluster_ids[j]]
        loss = -torch.log(my_preds).mean()
        print(loss)
        breakpoint()
