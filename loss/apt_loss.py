import torch
from sbi.utils import (
    clamp_and_warn,
    repeat_rows,
)

def apt_loss(self, round, training_theta, training_x, training_mask=None, use_non_atomic_loss=False):
    if round == 0:
        # Use posterior log prob (without proposal correction) for first round.
        #print("flow model forward calculation")
        log_prob = self.netPosterior.log_prob(training_theta, training_x)
    else:
        log_prob = _log_prob_proposal_posterior(self, training_theta, training_x, training_mask, use_non_atomic_loss)
    return - log_prob

def _log_prob_proposal_posterior(self,
        theta, x, masks=None, use_non_atomic_loss=False):
    """
    Return the log-probability of the proposal posterior.

    If the proposal is a MoG, the density estimator is a MoG, and the prior is
    either Gaussian or uniform, we use non-atomic loss. Else, use atomic loss (which
    suffers from leakage).

    Args:
        theta: Batch of parameters θ.
        x: Batch of data.
        masks: Mask that is True for prior samples in the batch in order to train
            them with prior loss.
        proposal: Proposal distribution.

    Returns: Log-probability of the proposal posterior.
    """

    if use_non_atomic_loss:
        return _log_prob_proposal_posterior_mog(theta, x)
    else:
        return _log_prob_proposal_posterior_atomic(self, theta, x, masks)

def _log_prob_proposal_posterior_atomic(self,
        theta, x, masks, _num_atoms = 10, _use_combined_loss = False
):
    """
    Return log probability of the proposal posterior for atomic proposals.

    We have two main options when evaluating the proposal posterior.
        (1) Generate atoms from the proposal prior.
        (2) Generate atoms from a more targeted distribution, such as the most
            recent posterior.
    If we choose the latter, it is likely beneficial not to do this in the first
    round, since we would be sampling from a randomly-initialized neural density
    estimator.

    Args:
        theta: Batch of parameters θ.
        x: Batch of data.
        masks: Mask that is True for prior samples in the batch in order to train
            them with prior loss.

    Returns:
        Log-probability of the proposal posterior.
    """

    batch_size = theta.shape[0]

    num_atoms = clamp_and_warn(
        "num_atoms", _num_atoms, min_val=2, max_val=batch_size
    )

    # Each set of parameter atoms is evaluated using the same x,
    # so we repeat rows of the data x, e.g. [1, 2] -> [1, 1, 2, 2]
    repeated_x = repeat_rows(x, num_atoms)

    # To generate the full set of atoms for a given item in the batch,
    # we sample without replacement num_atoms - 1 times from the rest
    # of the theta in the batch.
    probs = torch.ones(batch_size, batch_size) * (1 - torch.eye(batch_size)) / (batch_size - 1)

    choices = torch.multinomial(probs, num_samples=num_atoms - 1, replacement=False)
    contrasting_theta = theta[choices]
    # We can now create our sets of atoms from the contrasting parameter sets
    # we have generated.
    atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(
        batch_size * num_atoms, -1
    )
    # Evaluate large batch giving (batch_size * num_atoms) log prob posterior evals.
    #print("flow model forward calculation ...")
    log_prob_posterior = self.netPosterior.log_prob(atomic_theta, repeated_x)
    log_prob_posterior = log_prob_posterior.reshape(batch_size, num_atoms)
    #print("log prob posterior : ", log_prob_posterior.mean())
    # Get (batch_size * num_atoms) log prob prior evals.
    log_prob_prior = torch.Tensor(self.prior.log_prob(atomic_theta)).to(self.args.device)
    log_prob_prior = log_prob_prior.reshape(batch_size, num_atoms)

    # Compute unnormalized proposal posterior.
    unnormalized_log_prob = log_prob_posterior - log_prob_prior

    # Normalize proposal posterior across discrete set of atoms.
    log_prob_proposal_posterior = unnormalized_log_prob[:, 0] - torch.logsumexp(
        unnormalized_log_prob, dim=-1
    )

    # XXX This evaluates the posterior on _all_ prior samples
    if _use_combined_loss:
        log_prob_posterior_non_atomic = self.netPosterior.log_prob(theta, x)
        masks = masks.reshape(-1)
        log_prob_proposal_posterior = (
                masks * log_prob_posterior_non_atomic + log_prob_proposal_posterior
        )

    return log_prob_proposal_posterior

def _log_prob_proposal_posterior_mog(theta, x):
    pass