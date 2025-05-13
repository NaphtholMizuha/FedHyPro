from typing import Protocol, Tuple

import bitarray as ba
import numpy as np
from Pyfhel import PyCtxt
import galois
import numpy as np
import torch
from protector.ss import Shamir
from protector.ckks import Ckks
from protector.dp import Dp, VaryingDp


class Protector(Protocol):
    def protect(self, x, mask=None):
        pass

    def recover(self, x_he, x_dp, mask=None):
        pass


class DpProtector:
    def __init__(self, eps, delta, d_min, thr, n_round):
        self.dp = Dp(eps=eps, delta=delta, d_min=d_min, thr=thr, n_round=n_round)

    def protect(self, x):
        return self.dp.perturb(x)

    def recover(self, x):
        return x
    
class LayerDpProtector:
    def __init__(self, eps, delta, d_min, thr, n_round):
        self.dp = {
            key: Dp(eps=eps, delta=delta, d_min=d_min, thr=value, n_round=n_round)
            for key, value in thr.items()
        }
        
    def perturb(self, update):
        for key, value in update.items():
            if 'bn' not in key:
                shape = value.shape
                flatten = value.flatten().to('cpu').numpy()
                flatten = self.dp[key].perturb(flatten)
                update[key] = torch.from_numpy(flatten).reshape(shape).to('cuda')
        return update


class VaryingDpProtector:
    def __init__(self, eps, delta, d_min, thr, n_round, decay):
        self.dp = {
            key: VaryingDp(eps=eps, delta=delta, d_min=d_min, thr=value, n_round=n_round, decay=decay)
            for key, value in thr.items()
        }
        self.rnd = 0

    def protect(self, x):
        self.rnd += 1
        for key, value in x.items():
            if 'bn' not in key:
                shape = value.shape
                flatten = value.flatten().to('cpu').numpy()
                flatten = self.dp[key].perturb(flatten, self.rnd)
                x[key] = torch.from_numpy(flatten).reshape(shape).to('cuda')
        return x

    def perturb(self, x):
        return self.protect(x)

    def recover(self, x):
        return x


class CkksProtector:
    def __init__(self):
        self.ckks = Ckks()

    def protect(self, x):
        return self.ckks.encrypt(x)

    def recover(self, x):
        return self.ckks.decrypt(x)


class HybridProtector:
    def __init__(self, eps, delta, d_min, thr, n_round):
        self.ckks = Ckks()
        self.dp = Dp(eps=eps, delta=delta, d_min=d_min, thr=thr, n_round=n_round)

    def encrypt(self, x: np.ndarray):
        return self.ckks.encrypt(x)

    def decrypt(self, x):
        return self.ckks.decrypt(x)

    def perturb(self, x: np.ndarray) -> np.ndarray:
        return self.dp.perturb(x)

    def protect(self, x: np.ndarray, mask=ba.bitarray):
        """
        divede and protect the input data x with CKKS and DP

        Args:
            x: input data
            mask: a bit array indicating which part of the input data is protected by HE
        Returns:
            a tuple of ciphertext and perturbed data
        """
        he_idcs = np.array([i for i, bit in enumerate(mask) if bit])
        x_he = x[he_idcs]
        x_dp = x
        x_dp[he_idcs] = 0
        x_dp = self.dp.perturb(x_dp)
        x_dp[he_idcs] = 0
        # print(f'norm of he_part: {np.linalg.norm(x_he):.4f}, dp_part: {np.linalg.norm(x_dp):.4f}')
        return self.ckks.encrypt(x_he), x_dp

    def recover(self, x_he, x_dp: np.ndarray, mask: ba.bitarray) -> np.ndarray:
        """
        recover the global data from the ciphertext and perturbed data

        Args:
            x_he: ciphertext of HE part
            x_dp: perturbed DP part
            mask: a bit array indicating which part of the input data is protected by HE
        """
        he_idcs = np.array(
            [i for i, bit in enumerate(mask) if bit]
        )  # bitmask to idx of HE part
        x_he = self.ckks.decrypt(x_he)
        x_he = np.array(x_he)
        x_dp[he_idcs] = x_he
        return x_dp

class StevenProtector:
    def __init__(self, n_client, s_dim, packet_dim, eps, delta, d_min, thr, n_round, sigma=10, q=31352833):
        self.n = n_client
        self.s_dim = s_dim
        self.packet_dim = packet_dim
        self.gf = galois.GF(q)
        self.shamir = Shamir(self.gf)
        self.sigma = sigma
        self.q = q
        self.dp = LayerDpProtector(eps, delta, d_min, thr, n_round)
        StevenProtector.a_pub = self.gf.Random([self.packet_dim, self.s_dim])
        
    def perturb(self, x):
        return self.dp.perturb(x)

    def pad(self, x):
        remainder = x.size % self.packet_dim
        padding = self.packet_dim - remainder if remainder != 0 else 0
        self.padding = padding
        x = (
            np.concatenate([x, np.zeros(padding)], axis=0)
            .reshape([-1, self.packet_dim])
            .astype(np.int32)
        )
        return self.gf(x)

    def unpad(self, x):
        x = x.flatten()
        x = x[: -self.padding]
        return np.array(x)

    def split_secret(self):
        return self.shamir.split(self.secret, self.n, self.n)

    def protect(self, x):
        self.secret = self.gf.Random(self.s_dim)
        x = self.pad(x)
        noise = (
            np.random.normal(0, self.sigma, self.packet_dim).round().astype(np.int32)
            % self.q
        )
        error = self.gf(noise)
        b = self.a_pub.dot(self.secret) + error
        return x + b

    def recover(self, x, shares):
        secret = self.shamir.combine(shares)
        x = x - self.a_pub.dot(secret)
        return self.unpad(x)
