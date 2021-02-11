import torch
from nflows import flows, transforms
from nflows import distributions as distributions_
from nflows.nn import nets
from sbi.utils.sbiutils import standardizing_net, standardizing_transform
from sbi.utils.torchutils import create_alternating_binary_mask


def neural_net_nsf(self, hidden_features, num_blocks, num_bins, xDim, thetaDim, batch_x=None, batch_theta=None, tail=3., bounded=False, embedding_net = torch.nn.Identity()) -> torch.nn.Module:
    """Builds NSF p(x|y).

    Args:
        batch_x: Batch of xs, used to infer dimensionality and (optional) z-scoring.
        batch_y: Batch of ys, used to infer dimensionality and (optional) z-scoring.
        z_score_x: Whether to z-score xs passing into the network.
        z_score_y: Whether to z-score ys passing into the network.
        hidden_features: Number of hidden features.
        num_transforms: Number of transforms.
        embedding_net: Optional embedding network for y.
        kwargs: Additional arguments that are passed by the build function but are not
            relevant for maf and are therefore ignored.

    Returns:
        Neural network.
    """

    basic_transform = [
            transforms.CompositeTransform(
                [
                    transforms.PiecewiseRationalQuadraticCouplingTransform(
                        mask=create_alternating_binary_mask(
                            features=xDim, even=(i % 2 == 0)
                        ).to(self.args.device),
                        transform_net_create_fn=lambda in_features, out_features: nets.ResidualNet(
                            in_features=in_features,
                            out_features=out_features,
                            hidden_features=hidden_features,
                            context_features=thetaDim,
                            num_blocks=2,
                            activation=torch.relu,
                            dropout_probability=0.,
                            use_batch_norm=False,
                        ),
                        num_bins=num_bins,
                        tails='linear',
                        tail_bound=tail,
                        apply_unconditional_transform=False,
                    ),
                    transforms.RandomPermutation(features=xDim, device=self.args.device),
                    transforms.LULinear(xDim, identity_init=True),
                ]
            )
            for i in range(num_blocks)
        ]

    transform = transforms.CompositeTransform(basic_transform).to(self.args.device)

    if batch_theta != None:
        if bounded:
            transform_bounded = transforms.Logit(self.args.device)
            if self.sim.min[0].item() != 0 or self.sim.max[0].item() != 1:
                transfomr_affine = transforms.PointwiseAffineTransform(shift=-self.sim.min / (self.sim.max - self.sim.min),
                                                                           scale=1./(self.sim.max - self.sim.min))
                transform = transforms.CompositeTransform([transfomr_affine, transform_bounded, transform])
            else:
                transform = transforms.CompositeTransform([transform_bounded, transform])
        else:
            transform_zx = standardizing_transform(batch_x)
            transform = transforms.CompositeTransform([transform_zx, transform])
        embedding_net = torch.nn.Sequential(standardizing_net(batch_theta), embedding_net)
        distribution = distributions_.StandardNormal((xDim,), self.args.device)
        neural_net = flows.Flow(self, transform, distribution, embedding_net=embedding_net).to(self.args.device)
    else:
        distribution = distributions_.StandardNormal((xDim,), self.args.device)
        neural_net = flows.Flow(self, transform, distribution).to(self.args.device)

    return neural_net