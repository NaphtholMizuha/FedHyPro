import numpy as np
from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import Composition
from typing import Protocol


class DpProtocol(Protocol):
    def perturb(self, x: np.ndarray) -> np.ndarray: ...


class Dp:
    def __init__(self, eps, delta, d_min, thr, n_round) -> None:
        """
        compute the amplitude of Gaussian noise for differential privacy
        """
        self.sensitivity = (2 * thr) / d_min
        self.sigma = np.sqrt(2 * n_round * (-np.log(delta))) * self.sensitivity / eps
        self.d_min = d_min
        self.thr = thr

    def clip(self, x: np.ndarray):
        """
        clip the input x
        """
        norm = np.linalg.norm(x)
        if norm > self.thr:
            return x * self.thr / norm
        else:
            return x

    def noise(self, x: np.ndarray):
        """
        noise the input x with Gaussian noise, whose amplitude is pre-computed in `__init__()`

        Args:
            x: input data
        Returns:
            perturbed data
        """
        noised = x + np.random.normal(0, self.sigma, x.shape)
        # print(f'sum: {np.sum(x):.2f} -> {np.sum(noised):.2f}')
        return noised

    def perturb(self, x: np.ndarray):
        """
        clip and noise the input x

        Args:
            x: input data
        Returns:
            perturbed data
        """
        return self.noise(self.clip(x))
    
class VaryingDp(Dp):
    def __init__(self, eps, delta, d_min, thr, n_round, decay) -> None:
        super().__init__(eps, delta, d_min, thr, n_round)
        self.decay = decay
        self.delta = delta
        self.eps = eps
        self.n_round = n_round
    

    def get_sigma(self, rnd):
        sigma = np.sqrt(2 * ((np.power(self.decay, 1-rnd) - self.decay + np.power(self.decay, rnd-self.n_round)) / (1 - self.decay)) * (-np.log(self.delta))) * self.sensitivity / self.eps
        return np.sqrt(sigma**2 * self.decay**(rnd-1))
    
    def noise(self, x: np.ndarray, rnd: int):
        """
        noise the input x with Gaussian noise, whose amplitude is pre-computed in `__init__()`

        Args:
            x: input data
        Returns:
            perturbed data
        """
        noised = x + np.random.normal(0, self.get_sigma(rnd), x.shape)
        return noised
    
    def perturb(self, x: np.ndarray, rnd: int):
        """
        clip and noise the input x

        Args:
            x: input data
        Returns:
            perturbed data
        """
        # print(f'rnd is {rnd}')
        return self.noise(self.clip(x), rnd)

class DynamicDp:
    def __init__(self, delta, d_min, thr, n_round, init_sigma) -> None:
        self.sigma = init_sigma
        self.delta = delta
        self.d_min = d_min
        self.thr = thr
        self.n_round = n_round
        self.mechanisms = []

    def clip(self, x: np.ndarray):
        """
        clip the input x with threshold `self.thr`
        """
        norm = np.linalg.norm(x)
        if norm > self.thr:
            return x * self.thr / norm
        else:
            return x

    def noise(self, x: np.ndarray):
        """
        noise the input x with Gaussian noise, whose amplitude is `self.sigma`
        """
        # BUG: sigma is a dict
        return x + np.random.normal(0, self.sigma, x.shape)

    def get_eps(self):
        compose = Composition()
        mech = compose(self.mechanisms, [1] * len(self.mechanisms))
        return mech.get_approxDP(self.delta)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def get_sigma(self):
        return self.sigma

    def perturb(self, x: np.ndarray):
        sensitivity = (2 * self.thr) / self.d_min
        x_perturbed = self.noise(self.clip(x))
        # add the new Gaussian mechanism to the list
        self.mechanisms.append(GaussianMechanism(self.sigma / sensitivity))
        return x_perturbed
