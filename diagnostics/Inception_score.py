import numpy as np
from sklearn import mixture
from scipy.stats import entropy

def Inception_score(self, round):
    if round == 0:
        self.gmm = mixture.GaussianMixture(
            n_components=self.args.numModes, covariance_type='diag')
        self.gmm.fit(self.real_)
    pred_probs = self.gmm.predict_proba(self.parent.teacher_theta)

    scores = []
    # Calculating the inception score
    part = pred_probs + 1e-6
    py = np.mean(part, axis=0)
    for i in range(part.shape[0]):
        pyx = part[i, :]
        scores.append(entropy(pyx, py))
    inception_score = np.exp(np.mean(scores))

    return inception_score