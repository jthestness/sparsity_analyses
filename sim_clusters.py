
import argparse
import numpy as np
import torch


class MLPModel(torch.nn.Module):
    def __init__(self, num_clusters, hidden_sizes):
        super(MLPModel, self).__init__()
        if type(hidden_sizes) == int:
            hidden_sizes = [hidden_sizes]
        hidden_sizes.append(num_clusters)
        self.hidden_layers = []
        for idx in range(len(hidden_sizes) - 1):
            self.my_weight = torch.nn.Linear(hidden_sizes[idx], hidden_sizes[idx+1])
            self.hidden_layers.append(self.my_weight)

    def forward(self, inputs):
        output = inputs
        for hidden_layer in self.hidden_layers:
            output = hidden_layer(output)
        return output


def generate_data(cluster_vectors, batch_size, data_dimension, num_clusters, noise_coefficient):
    # GENERATOR #2: Generates random points on Gaussian unit-ball.
    # Cluster ID is the cluster_vector with largest dot-product.
    # This is also poor, because it samples random points on the Gaussian
    # unit-ball and those points could be far from any cluster vectors
    batch_vectors = np.random.normal(size=(batch_size, data_dimension)) / np.sqrt(data_dimension)
    mapping = np.matmul(batch_vectors, np.transpose(cluster_vectors))
    cluster_ids = torch.tensor(mapping.argmax(axis=-1))
    batch_vectors = torch.tensor(batch_vectors, dtype=torch.float)
    return batch_vectors, cluster_ids

    # GENERATOR #1: Generates random points on Gaussian ball around cluster_vectors
    # This is poor, because it typically generates spheres around points, so sampling is
    # dense only in noise_coefficient-sized spheres around each cluster_vector
    cluster_ids = np.random.choice(np.arange(num_clusters), size=(batch_size))
    batch_vectors = cluster_vectors[cluster_ids]
    noise_vectors = np.random.normal(size=(batch_size, data_dimension)) * noise_coefficient
    batch_vectors += noise_vectors
    batch_vectors = torch.tensor(batch_vectors, dtype=torch.float)
    cluster_ids = torch.tensor(cluster_ids)
    return batch_vectors, cluster_ids

def run_eval(step_idx, my_model, cluster_vectors, batch_size, data_dimension, num_clusters, noise_coefficient):
    total_loss = None
    num_eval_iters = 100
    for i in range(num_eval_iters):
        batch_vectors, cluster_ids = generate_data(cluster_vectors, batch_size, data_dimension, num_clusters, noise_coefficient)

        my_outs = my_model(batch_vectors)
        loss = my_loss(my_outs, cluster_ids)
        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss
    total_loss /= num_eval_iters
    print(f'Eval loss, {step_idx}: {total_loss.item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dimension', type=int, default=64)
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--noise_coefficient', type=float, default=0.1)
    parser.add_argument('--num_steps', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    data_dimension = args.data_dimension
    num_clusters = args.num_clusters
    hidden_size = args.hidden_size
    noise_coefficient = args.noise_coefficient
    num_steps = args.num_steps
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    cluster_vectors = np.random.normal(size=(num_clusters, data_dimension)) / np.sqrt(data_dimension)
    clust_vecs = cluster_vectors / np.linalg.norm(cluster_vectors, axis=-1)[:,None]
    prod = np.matmul(clust_vecs, np.transpose(clust_vecs)) - np.eye(num_clusters)
    print(f'Centers alignment mean: {prod.mean()}')
    print(f'Centers alignment stdev: {prod.std()}')
    print(f'Centers malignment (+): {(np.abs(prod).max() - prod.mean()) / prod.std():.2f}')
    print(f'Centers malignment (-): {(prod.min() - prod.mean()) / prod.std():.2f}')

    breakpoint()

    my_model = MLPModel(hidden_sizes=[data_dimension, hidden_size], num_clusters=num_clusters)
    my_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    print(f'{batch_size} {num_steps} {noise_coefficient} {data_dimension} {hidden_size} {num_clusters}')
    for i in range(num_steps):
        # Generate randomized data vectors
        batch_vectors, cluster_ids = generate_data(cluster_vectors, batch_size, data_dimension, num_clusters, noise_coefficient)

        my_outs = my_model(batch_vectors)
        loss = my_loss(my_outs, cluster_ids)
        if i % (num_steps // 100) == 0:
            print(f'{i * 100 / num_steps}%: {loss.item()}', flush=True)
        # print(loss)
        loss.backward()
        optimizer.step()
        # breakpoint()

        if i % (num_steps // 10) == 0:
            run_eval(i, my_model, cluster_vectors, batch_size, data_dimension, num_clusters, noise_coefficient)

    run_eval(num_steps, my_model, cluster_vectors, batch_size, data_dimension, num_clusters, noise_coefficient)
