import numpy as np
import torch

def theta_domain(args, simulation):
    thetaDomain = torch.Tensor([[simulation.min[0].item(), simulation.max[0].item()]])
    for i in range(1, args.thetaDim):
        thetaDomain = torch.cat(
            (thetaDomain, torch.Tensor([[simulation.min[i].item(), simulation.max[i].item()]])))
    return thetaDomain.cpu().detach().numpy()

def test_thetas(args, simulation):
    thetaDomain = theta_domain(args, simulation)
    test_thetas = []
    num = 501
    lin = np.linspace(0, 1, num)
    base = np.linspace(0,1,num).reshape(-1,1)
    for j in range(num):
        for i in range(num):
            test_thetas.append([thetaDomain[0][0] + (thetaDomain[0][1] - thetaDomain[0][0]) * lin[i],
                                thetaDomain[1][0] + (thetaDomain[1][1] - thetaDomain[1][0]) * lin[j]])
    test_thetas = torch.Tensor(test_thetas).to(args.device).to(args.device)

    xx, yy = np.meshgrid(thetaDomain[0][0] + (thetaDomain[0][1] - thetaDomain[0][0]) * lin,
                                   thetaDomain[1][0] + (thetaDomain[1][1] - thetaDomain[1][0]) * lin)

    return test_thetas, xx, yy