import matplotlib.pyplot as plt
import torch
import os
import numpy as np

def plotLikelihood(args, netLikelihood, observation, test_thetas, xx, yy, dir, round):
    num = np.sqrt(test_thetas.shape[0])
    plt.close()
    estimatedLikelihood = torch.exp(
        netLikelihood.log_prob
            (inputs=observation[:args.xDim].repeat(test_thetas.shape[0], 1), context=test_thetas.to(args.device))).detach()
    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, estimatedLikelihood.cpu().numpy().reshape(num, num),
                 100, cmap=plt.cm.gray)# cmap='binary')

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(dir + '/likelihood_' + str(round) + '.pdf')
    plt.close()