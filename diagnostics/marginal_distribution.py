import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
from diagnostics.base import theta_domain
from diagnostics.groundtruth_inputs import get_groundtruth_inputs

def plot_marginal_distribution(args, bases, simulation, round, input_theta, title, upper=True):
    plt.close()
    fig = plt.figure(figsize=(16 ,16))
    thetaDomain = theta_domain(args, simulation)
    true_thetas = get_groundtruth_inputs(args)
    for i in range(args.thetaDim):
        for j in range(i, args.thetaDim) if upper else range(i + 1):

            ax = fig.add_subplot(args.thetaDim, args.thetaDim, i * args.thetaDim + j + 1)

            if i == j:
                bandwidths = 10 ** np.linspace(-4, 0, 30)
                grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                    {'bandwidth': bandwidths},
                                    cv=10)
                grid.fit(input_theta[: ,j].reshape(-1 ,1))
                kde = grid.best_estimator_
                likelihood = np.exp(kde.score_samples(bases[j])).reshape(-1)
                ax.plot(bases[j], likelihood)
                ax.set_xlim(thetaDomain[j])
                ax.set_ylim([0.0, ax.get_ylim()[1]])
                ax.set_ylim([0.0, ax.get_ylim()[1]])
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                if i < args.thetaDim - 1 and not upper:
                    ax.tick_params(axis='x', which='both', labelbottom=False)
                if true_thetas.shape[0] > 1:
                    for k in range(true_thetas.shape[0]):
                        ax.vlines(true_thetas[k][j].item(), 0, ax.get_ylim()[1], color='r')
                else:
                    if true_thetas is not None: ax.vlines(true_thetas[0][j].item(), 0, ax.get_ylim()[1], color='r')

            else:
                ax.scatter(input_theta[: ,j], input_theta[: ,i], s=1, color='black')
                ax.set_xlim(thetaDomain[j])
                ax.set_ylim(thetaDomain[i])
                if i < args.thetaDim - 1: ax.tick_params(axis='x', which='both', labelbottom=False)
                if j > 0: ax.tick_params(axis='y', which='both', labelleft=False)
                if j == args.thetaDim - 1: ax.tick_params(axis='y', which='both', labelright=True)
                if true_thetas.shape[0] > 1:
                    for k in range(true_thetas.shape[0]):
                        ax.plot(true_thetas[k][j].item(), true_thetas[k][i].item(), 'r.', ms=8)
                else:
                    if true_thetas is not None: ax.plot(true_thetas[0][j].item(), true_thetas[0][i].item(), 'r.', ms=8)
    plt.tight_layout()
    plt.close()