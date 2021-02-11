import sbi.utils.metrics as metrics
from sampler.sample_from_exact_likelihood import sample_from_exact_likelihood

def Maximum_mean_discrepancy(parent, real, fake, round, wxs=None, wys=None, scale=None, return_scale=False):
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
        real = sample_from_exact_likelihood(parent, 5000).cpu().detach()

    mmd2 = metrics.unbiased_mmd_squared(real, fake.cpu().detach()).item()

    return mmd2