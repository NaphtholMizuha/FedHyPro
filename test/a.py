import galois
import numpy as np
n_client = 10
len_secret = 256
dim = 10
q = 2097169
gf = galois.GF(q, repr='int')
A = gf.Random((dim, len_secret))
grad = {}
secret = {}
cipher = {}

for i in range(n_client):
    grad[i] = gf.Random(dim)
    secret[i] = gf.Random(len_secret)
    noise = np.random.normal(0, 3, dim).round().astype(int) % q
    error = gf(noise)
    b = A.dot(secret[i]) + error
    cipher[i] = grad[i] + b



grad_sum = sum(grad.values(), start=gf.Zeros(dim))
secret_sum = sum(secret.values(), start=gf.Zeros(len_secret))
cipher_sum = sum(cipher.values(), start=gf.Zeros(dim))

recover = cipher_sum - A.dot(secret_sum)
print(recover)
print(grad_sum)
