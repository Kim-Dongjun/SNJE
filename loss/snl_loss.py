

def snl_loss(self, theta, x):
    loss = - self.netLikelihood.log_prob(context=theta, inputs=x).mean()
    return loss