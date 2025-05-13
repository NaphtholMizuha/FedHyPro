import galois
import numpy as np
from dataclasses import dataclass

@dataclass
class Share:
    x: galois.FieldArray
    y: galois.FieldArray
    
    def __add__(self, other):
        return Share(self.x, self.y + other.y)

class Shamir:
    def __init__(self, gf):
        self.gf = gf

    def split(self, secret: galois.FieldArray, n: int, k: int):
        coeffs = [np.insert(self.gf.Random(k - 1), k-1, s) for s in secret]
        shares = []
        for i in range(1, n + 1):
            x = self.gf(i)
            y = [galois.Poly(coeff, field=self.gf)(x) for coeff in coeffs]
            y = self.gf(y)
            shares.append(Share(x, y))
            
        return shares
    
    def combine(self, shares: list[Share]):
        n_secret = len(shares[0].y)
        secret = self.gf([0 for _ in range(n_secret)])
        
        for i in range(len(shares)):
            x_i, y_i = shares[i].x, shares[i].y
            l_i = self.gf(1)
            
            for j in range(len(shares)):
                if i != j:
                    x_j = shares[j].x
                    l_i *= x_j / (x_j - x_i)
            secret += y_i * l_i
            
        return secret