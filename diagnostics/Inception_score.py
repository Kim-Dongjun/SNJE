import numpy as np
from sklearn import mixture
from scipy.stats import entropy

def Inception_score(args, groundtruth_sample, sample, round, gmm=None):
    if round == 0:
        gmm = mixture.GaussianMixture(
            n_components=args.numModes, covariance_type='diag')
        gmm.fit(groundtruth_sample)
    pred_probs = gmm.predict_proba(sample)

    scores = []
    # Calculating the inception score
    part = pred_probs + 1e-6
    py = np.mean(part, axis=0)
    for i in range(part.shape[0]):
        pyx = part[i, :]
        scores.append(entropy(pyx, py))
    inception_score = np.exp(np.mean(scores))

    return gmm, inception_score