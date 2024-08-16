
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', type=int, default=8)
    parser.add_argument('--data_dimension', type=int, default=3)
    parser.add_argument('--unit_grads', action="store_true")
    args = parser.parse_args()

    clust_vecs = torch.normal(torch.tensor(0.0), torch.sqrt(torch.tensor(1.0 / args.data_dimension)), size=(args.num_clusters, args.data_dimension))

    norm_cv = clust_vecs / torch.linalg.norm(clust_vecs, axis=-1, keepdims=True)

    for i in range(10):
        print(f'Step: {i}')
        # breakpoint()

        dotprod = torch.matmul(norm_cv, torch.transpose(norm_cv, 0, 1))
        smdot = torch.softmax(dotprod - 1000.0 * torch.eye(args.num_clusters), axis=-1)
        cv_extend = norm_cv[None,:,:]
        diffs = cv_extend - torch.transpose(cv_extend, 0, 1)
        norm_cv_grad = (diffs * smdot[:,:,None]).sum(axis=0)
        if args.unit_grads:
            norm_cv_grad /= torch.linalg.norm(norm_cv_grad, axis=-1, keepdims=True)

        min_step_size = 0.00001
        step_size = 0.00001
        max_step_size = 1024.0
        best_loss = None
        step_taken = False
#        while not step_taken:
#         for step_size in [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10., 30., 100., 300., 1000.]:
        for step_size in [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003]:
            upd_norm_cv = norm_cv + step_size * norm_cv_grad
            upd_norm_cv /= torch.linalg.norm(upd_norm_cv, axis=-1, keepdims=True)
            dotprod = torch.matmul(upd_norm_cv, torch.transpose(upd_norm_cv, 0, 1))
            if best_loss is None:
                best_loss = dotprod.sum()
                margin = torch.abs(best_loss) * 0.0001
            else:
                if dotprod.sum() < best_loss + margin and dotprod.sum() > best_loss - margin:
                    norm_cv = upd_norm_cv
                    step_taken = True
                if dotprod.sum() > best_loss:
                    max_step_size = step_size
                else:
                    best_loss = dotprod.sum()
                    margin = torch.abs(best_loss) * 0.0001
                    min_step_size = step_size
            print(f'    {step_size} {dotprod.sum().item()} {best_loss}')
            step_size = (max_step_size + min_step_size) / 2.0

        angle = torch.arccos(torch.max(dotprod, torch.tensor(-1.0 + 1e-7)))
        # print(f'\n  {dotprod}\n')
        angle = torch.where(torch.isnan(angle), 0.0, angle) + 1000.0 * torch.eye(args.num_clusters)
        print(f'  Min angle: {torch.rad2deg(angle).min()}\n')
        # print(f'  {norm_cv}\n')

#    breakpoint()


