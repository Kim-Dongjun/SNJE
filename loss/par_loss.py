from loss.apt_loss import apt_loss


def par_loss(self, theta, mask):

    fake_x = self.netLikelihood.sample(1, context=theta).reshape(theta.shape[0], -1)

    par_estimation = apt_loss(self, round, theta, fake_x, mask,
                                    self.args.use_non_atomic_loss).mean()

    return par_estimation