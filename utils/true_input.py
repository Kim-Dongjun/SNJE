import torch

def get_true_input(args):
    if args.simulation == 'SLCP-16':
        return torch.Tensor([[1.5, -2.0, -1., -0.9, 0.6]]).to(args.device)

    elif args.simulation == 'SLCP-256':
        return torch.Tensor([[1.5, 2.0, 1.3, 1.2, 1.8, 2.5, 1.6, 1.1]]).to(args.device)

    elif args.simulation == 'MG1':
        return torch.Tensor([[1, 4, 0.2]]).to(args.device)

    elif args.simulation == 'Ricker':
        return torch.Tensor([[3.8,0.3,10.]]).to(args.device)

    elif args.simulation == 'Poisson':
        return torch.Tensor([[0.1, 0.9, 0.6, 0.2]]).to(args.device)

    elif args.simulation == 'Cosine':
        return torch.zeros(args.thetaDim).reshape(1, -1).to(args.device)

    elif args.simulation == 'MNIST':
        return torch.Tensor([[0.5, 0.5, 0.5, 0.5, 0.5]]).to(args.device)

    elif args.simulation == 'FashionMNIST':
        return torch.Tensor([[0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9]]).to(args.device)

    elif args.simulation == 'SVHN':
        return torch.Tensor([[0.6818, 0.5421, 0.6628, 0.6103, 0.1689, 0.1241, 0.7385, 0.6024, 0.4593]]).to(args.device)
