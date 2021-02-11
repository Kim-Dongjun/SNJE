"""Implementations of permutation-like transforms."""

import torch

from nflows.transforms.base import Transform
import nflows.utils.typechecks as check


class Permutation(Transform):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation, dim=1):
        if permutation.ndimension() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not check.is_positive_int(dim):
            raise ValueError("dim must be a positive integer.")

        super().__init__()
        self._dim = dim
        self.register_buffer("_permutation", permutation)
        if self._permutation.get_device() == -1:
            self.device = 'cpu'
        else:
            self.device = 'cuda:'+str(int(self._permutation.get_device()))

    @property
    def _inverse_permutation(self):

        return torch.argsort(self._permutation).to(self.device)

    @staticmethod
    def _permute(inputs, permutation, dim, device):
        if dim >= inputs.ndimension():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError(
                "Dimension {} in inputs must be of size {}.".format(
                    dim, len(permutation)
                )
            )
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation).to(device)
        logabsdet = torch.zeros(batch_size).to(device)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._permute(inputs, self._permutation, self._dim, self.device)

    def inverse(self, inputs, context=None):
        return self._permute(inputs, self._inverse_permutation, self._dim, self.device)


class RandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features, device, dim=1):
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.randperm(features).to(device), dim)


class ReversePermutation(Permutation):
    """Reverses the elements of the input. Only works with 1D inputs."""

    def __init__(self, features, dim=1):
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.arange(features - 1, -1, -1), dim)
