import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def negative_log_probability(teacher_theta, true_thetas):
    bandwidths = 10 ** np.linspace(-4, 1, 30)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=10)
    grid.fit(teacher_theta)
    kde = grid.best_estimator_
    negative_log_probability = - np.sum(kde.score_samples(true_thetas.cpu().detach().numpy()))

    return negative_log_probability