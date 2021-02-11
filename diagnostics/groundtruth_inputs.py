import numpy as np
import torch

def get_groundtruth_inputs(args):
    if args.simulation == 'SLCP-16':
        true_thetas = torch.Tensor(
            [[1.5, -2.0, -1.0, -0.9, 0.6], [1.5, -2.0, -1.0, 0.9, 0.6], [1.5, -2.0, 1.0, -0.9, 0.6],
             [1.5, -2.0, 1.0, 0.9, 0.6],
             [-1.5, -2.0, -1.0, -0.9, 0.6], [-1.5, -2.0, -1.0, 0.9, 0.6], [-1.5, -2.0, 1.0, -0.9, 0.6],
             [-1.5, -2.0, 1.0, 0.9, 0.6],
             [1.5, 2.0, -1.0, -0.9, 0.6], [1.5, 2.0, -1.0, 0.9, 0.6], [1.5, 2.0, 1.0, -0.9, 0.6],
             [1.5, 2.0, 1.0, 0.9, 0.6],
             [-1.5, 2.0, -1.0, -0.9, 0.6], [-1.5, 2.0, -1.0, 0.9, 0.6], [-1.5, 2.0, 1.0, -0.9, 0.6],
             [-1.5, 2.0, 1.0, 0.9, 0.6]]).to(args.device)

    elif args.simulation == 'SLCP-256':
        true_thetas = np.array([1.5, 2.0, 1.3, 1.2, 1.8, 2.5, 1.6, 1.1])
        from itertools import chain, combinations

        def powerset(set):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            return chain.from_iterable(combinations(set, r) for r in range(len(set) + 1))

        fullList = np.arange(true_thetas.shape[0]).tolist()
        subsets = list(powerset(np.arange(true_thetas.shape[0])))
        for subset in subsets:
            subset = list(subset)
            theta = []
            for k in range(true_thetas.shape[0]):
                if k in subset:
                    theta.append(true_thetas[k])
                else:
                    theta.append(-true_thetas[k])
            if len(subset) == 0:
                true_thetas = torch.Tensor(theta).reshape(1, -1)
            else:
                true_thetas = torch.cat((true_thetas, torch.Tensor(theta).reshape(1, -1)))
        true_thetas = true_thetas.to(args.device)

    elif args.simulation == 'shubert':
        true_thetas = torch.Tensor(
            [[-7.090000152587891, -7.710000038146973], [-7.710000038146973, -7.090000152587891],
             [-0.8100000023841858, -7.710000038146973], [-1.4299999475479126, -7.090000152587891],
             [-0.8100000023841858, -1.440000057220459], [-1.440000057220459, -0.8100000023841858],
             [-7.090000152587891, -1.440000057220459], [-7.710000038146973, -0.8100000023841858],
             [4.849999904632568, -0.8100000023841858], [-7.090000152587891, 4.849999904632568],
             [-0.8100000023841858, 4.849999904632568], [5.46999979019165, 4.849999904632568],
             [-7.71999979019165, 5.46999979019165], [-1.4299999475479126, 5.46999979019165],
             [4.849999904632568, 5.46999979019165], [5.46999979019165, -7.71999979019165],
             [4.860000133514404, -7.099999904632568], [5.480000019073486, -1.440000057220459]]).to(self.args.device)

    elif args.simulation == 'multiModes':
        true_thetas = torch.Tensor([[0.1, 0.1],
                                         [0.3, 0.1],
                                         [0.5, 0.1],
                                         [0.7, 0.1],
                                         [0.9, 0.1],
                                         [0.1, 0.3],
                                         [0.3, 0.3],
                                         [0.5, 0.3],
                                         [0.7, 0.3],
                                         [0.9, 0.3],
                                         [0.1, 0.5],
                                         [0.3, 0.5],
                                         [0.5, 0.5],
                                         [0.7, 0.5],
                                         [0.9, 0.5],
                                         [0.1, 0.7],
                                         [0.3, 0.7],
                                         [0.5, 0.7],
                                         [0.7, 0.7],
                                         [0.9, 0.7],
                                         [0.1, 0.9],
                                         [0.3, 0.9],
                                         [0.5, 0.9],
                                         [0.7, 0.9],
                                         [0.9, 0.9]]).to(args.device)

    elif args.simulation == 'mg1':
        true_thetas = torch.Tensor([[1, 4, 0.2]]).to(args.device)

    elif args.simulation == 'ricker':
        true_thetas = torch.Tensor([[3.8 ,0.3 ,10.]]).to(args.device)

    elif args.simulation == 'poisson':
        true_thetas = torch.Tensor([[0.1, 0.9, 0.6, 0.2]]).to(args.device)

    elif args.simulation == 'mnist':
        true_thetas = torch.Tensor([[0.5, 0.5, 0.5, 0.5, 0.5]]).to(args.device)

    elif args.simulation == 'fashion-mnist':
        true_thetas = torch.Tensor([[0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]]).to(args.device)

    elif args.simulation == 'svhn':
        true_thetas = torch.Tensor([[0.6818, 0.5421, 0.6628, 0.6103, 0.1689, 0.1241, 0.7385, 0.6024, 0.4593]]).to(args.device)

    return true_thetas