import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from tqdm.auto import tqdm
import logging

def sample_from_posterior(self, num_samples,
    init=False,
    show_progress_bars: bool = False,
    warn_acceptance: float = 0.01,
    sample_for_correction_factor: bool = False,
):
    # This code is comming from SBI package (https://github.com/mackelab/sbi)
    r"""Return samples from a posterior $p(\theta|x)$ only within the prior support.

    This is relevant for snpe methods and flows for which the posterior tends to have
     mass outside the prior boundaries.

    This function uses rejection sampling with samples from posterior in order to
        1) obtain posterior samples within the prior support, and
        2) calculate the fraction of accepted samples as a proxy for correcting the
           density during evaluation of the posterior.

    Args:
        posterior_nn: Neural net representing the posterior.
        prior: Distribution-like object that evaluates probabilities with `log_prob`.
        x: Conditioning variable $x$ for the posterior $p(\theta|x)$.
        num_samples: Desired number of samples.
        show_progress_bars: Whether to show a progressbar during sampling.
        warn_acceptance: A minimum acceptance rate under which to warn about slowness.
        sample_for_correction_factor: True if this function was called by
            `leakage_correction()`. False otherwise. Will be used to adapt the leakage
             warning.

    Returns:
        Accepted samples and acceptance rate as scalar Tensor.
    """

    if init:
        thetas = torch.Tensor(self.prior.gen(num_samples)).to(self.args.device).detach()
        return thetas
    else:
        self.netPosterior.eval()
        #assert not self.netPosterior.training, "Posterior nn must be in eval mode for sampling."

        # Progress bar can be skipped, e.g. when sampling after each round just for logging.
        pbar = tqdm(
            disable=not show_progress_bars,
            total=num_samples,
            desc=f"Drawing {num_samples} posterior samples",
        )

        num_remaining, num_sampled_total = num_samples, 0
        accepted, acceptance_rate = [], float("Nan")
        leakage_warning_raised = False
        # In each iteration of the loop we sample the remaining number of samples from the
        # posterior. Some of these samples have 0 probability under the prior, i.e. there
        # is leakage (acceptance rate<1) so sample again until reaching `num_samples`.
        #print("sampling from posterior ...")
        while num_remaining > 0:

            candidates = self.netPosterior.sample(num_remaining, context=self.observation)

            # TODO we need this reshape here because posterior_nn.sample sometimes return
            # leading singleton dimension instead of (num_samples), e.g., (1, 10000, 4)
            # instead of (10000, 4). This can't be handled by MultipleIndependent, see #141.
            candidates = candidates.reshape(num_remaining, -1)
            num_sampled_total += num_remaining

            are_within_prior = torch.isfinite(self.prior.log_prob(candidates))
            accepted.append(candidates[are_within_prior])

            num_accepted = are_within_prior.sum().item()
            pbar.update(num_accepted)
            num_remaining -= num_accepted

            # To avoid endless sampling when leakage is high, we raise a warning if the
            # acceptance rate is too low after the first 1_000 samples.
            acceptance_rate = (num_samples - num_remaining) / num_sampled_total
            if (
                num_sampled_total > 1000
                and acceptance_rate < warn_acceptance
                and not leakage_warning_raised
            ):
                if sample_for_correction_factor:
                    logging.warning(
                        f"""Drawing samples from posterior to estimate the normalizing
                            constant for `log_prob()`. However, only {acceptance_rate:.0%}
                            posterior samples are within the prior support. It may take a
                            long time to collect the remaining {num_remaining} samples.
                            Consider interrupting (Ctrl-C) and either basing the estimate
                            of the normalizing constant on fewer samples (by calling
                            `posterior.leakage_correction(x_o, num_rejection_samples=N)`,
                            where `N` is the number of samples you want to base the
                            estimate on (default N=10000), or not estimating the
                            normalizing constant at all
                            (`log_prob(..., norm_posterior=False)`. The latter will result
                            in an unnormalized `log_prob()`."""
                    )
                else:
                    logging.warning(
                        f"""Only {acceptance_rate:.0%} posterior samples are within the
                            prior support. It may take a long time to collect the remaining
                            {num_remaining} samples. Consider interrupting (Ctrl-C)
                            and switching to `sample_with_mcmc=True`."""
                    )
                leakage_warning_raised = True  # Ensure warning is raised just once.

        pbar.close()
        self.netPosterior.train()
        return torch.cat(accepted)#, as_tensor(acceptance_rate)