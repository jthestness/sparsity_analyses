
import argparse
import numpy as np

def get_random_mask(size=None, sparsity=None):
    mask = np.random.normal(size=size)
    sort_mask = np.sort(mask)
    min = int(mask.size * sparsity)
    return np.where(mask > sort_mask[min], 1.0, 0.0)

def get_topk_mask(array=None, sparsity=None):
    sort_mask = np.sort(np.abs(array))
    min = int(array.size * sparsity)
    return np.where(array > sort_mask[min], array, 0.0) + np.where(array < -sort_mask[min], array, 0.0)

def angle_between(array1, array2):
    dot = np.dot(array1, array2)
    return np.degrees(np.arccos(dot / (np.linalg.norm(array1) * np.linalg.norm(array2))))


parser = argparse.ArgumentParser()
parser.add_argument('--random_grad', action='store_true')
parser.add_argument('--batch_size', default=1, type=int)
args = parser.parse_args()

hidden_size = 1024
num_params = hidden_size * hidden_size

if args.random_grad:
    fake_grad = np.random.normal(size=(num_params, args.batch_size)).sum(axis=-1)
    fake_grad /= np.sqrt(args.batch_size)
else:
    fake_dy = np.random.normal(size=(hidden_size, args.batch_size))
    fake_x = np.random.normal(size=(hidden_size, args.batch_size))
    fake_grad = np.matmul(fake_dy, np.transpose(fake_x)) / np.sqrt(args.batch_size)
    fake_grad = fake_grad.flatten()

len_grad = np.linalg.norm(fake_grad)
print(f'Length of gradient: {len_grad}')

sparsity_levels = [1.0 - 2**(-(i+1)/2) for i in range(28)]

for sparsity in sparsity_levels:
#    print(f'Sparsity level {sparsity}')
    for idx in range(1):
        mask = get_random_mask(size=(num_params), sparsity=sparsity)
        fake_mask_grad = fake_grad * mask
        rand_sparsity = (fake_mask_grad == 0.0).sum() / fake_mask_grad.size
        len_rand_mask_grad = np.linalg.norm(fake_mask_grad)
        print(f'  {idx}: Length of rand mask grad: {len_rand_mask_grad}')
        rand_angle = angle_between(fake_mask_grad, fake_grad)
        print(f'  {idx}: Angle to rand mask grad: {rand_angle}')

        fake_mask_grad = get_topk_mask(array=fake_grad, sparsity=sparsity)
        topk_sparsity = (fake_mask_grad == 0.0).sum() / fake_mask_grad.size
        len_topk_mask_grad = np.linalg.norm(fake_mask_grad)
        print(f'  {idx}: Length of rand mask grad: {len_topk_mask_grad}')
        topk_angle = angle_between(fake_mask_grad, fake_grad)
        print(f'  {idx}: Angle to topk mask grad: {topk_angle}')
        print(f'{sparsity}\t{rand_sparsity}\t{len_rand_mask_grad}\t{rand_angle}\t{topk_sparsity}\t{len_topk_mask_grad}\t{topk_angle}')

# breakpoint()
