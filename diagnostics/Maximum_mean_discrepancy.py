import sbi.utils.metrics as metrics
from sampler.sample_from_exact_likelihood import sample_from_exact_likelihood

def Maximum_mean_discrepancy(self, groundtruth_sample, sample, round, wxs=None, wys=None, scale=None, return_scale=False):
    """
    Finite sample estimate of square maximum mean discrepancy. Uses a gaussian kernel.
    :param xs: first sample
    :param ys: second sample
    :param wxs: weights for first sample, optional
    :param wys: weights for second sample, optional
    :param scale: kernel scale. If None, calculate it from data
    :return: squared mmd, scale if not given
    """
    if round == 0:
        groundtruth_sample = sample_from_exact_likelihood(self, 5000).cpu().detach()

    mmd2 = metrics.unbiased_mmd_squared(groundtruth_sample, sample.cpu().detach()).item()

    return groundtruth_sample, mmd2