

def get_simulator(args):

    if args.simulation == 'SLCP-16':
        from simulators.SLCP_16 import SLCP_16
        simulator = SLCP_16(args)

    elif args.simulation == 'SLCP-256':
        from simulators.SLCP_256 import SLCP_256
        simulator = SLCP_256(args)

    elif args.simulation == 'MG1':
        from simulators.MG1 import MG1
        simulator = MG1(args)

    elif args.simulation == 'Ricker':
        from simulators.Ricker import Ricker
        simulator = Ricker(args)

    elif args.simulation == 'poisson':
        from simulators.Poisson import Poisson
        simulator = Poisson(args)

    elif args.simulation in ['MNIST', 'FashionMNIST', 'SVHN']:
        from simulators.Image import Image
        simulator = Image(args)

    elif args.simulation == 'Cosine':
        from simulators.Cosine import Cosine
        simulator = Cosine(args)

    return simulator