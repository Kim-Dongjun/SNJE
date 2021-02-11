import torch

def get_input_range(args):

    if args.simulation == 'SLCP-16':
        min = torch.Tensor([-3.] * args.thetaDim).to(args.device)
        max = torch.Tensor([3.] * args.thetaDim).to(args.device)

    elif args.simulation == 'SLCP-256':
        min = torch.Tensor([-3.] * args.thetaDim).to(args.device)
        max = torch.Tensor([3.] * args.thetaDim).to(args.device)

    elif args.simulation == 'MG1':
        min = torch.Tensor([0., 0., 0.]).to(args.device)
        max = torch.Tensor([10., 10., 1. / 3.]).to(args.device)

    elif args.simulation == 'Cosine':
        min = torch.Tensor([0.] * args.thetaDim).to(args.device)
        max = torch.Tensor([1.] * args.thetaDim).to(args.device)

    elif args.simulation == 'Ricker':
        min = torch.Tensor([0., 0., 0.]).to(args.device)
        max = torch.Tensor([5., 1., 15.]).to(args.device)

    elif args.simulation in ['Poisson', 'MNIST', 'FashionMNIST', 'SVHN']:
        min = torch.Tensor([0.] * args.thetaDim).to(args.device)
        max = torch.Tensor([1.] * args.thetaDim).to(args.device)

    return min, max