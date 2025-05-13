import numpy as np
import tenseal as ts
from utils.timer import suppress_output


class Ckks:
    def __init__(self):
        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        self.ctx.global_scale = 2**40
        self.ctx.generate_galois_keys()

    def encrypt(self, data: np.ndarray) -> ts.CKKSVector:
        with suppress_output():
            cipher = ts.ckks_vector(self.ctx, data)
        return cipher

    def decrypt(self, cipher: ts.CKKSVector) -> np.ndarray:
        return cipher.decrypt()
