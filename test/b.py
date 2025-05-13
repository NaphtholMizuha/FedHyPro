import galois

gf = galois.GF(2**8)

a = gf.Random(100000000).reshape([1000, -1])
print(a.size)