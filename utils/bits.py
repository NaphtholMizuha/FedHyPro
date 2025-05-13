import bitarray as ba
import numpy as np
from utils.timer import timer

def arr_to_bits(arr, length):
    with timer("arr_to_bits") as t:
        bits = ba.bitarray(length)
        for idx in arr:
            bits[idx] = True
    print(f"arr_to_bits took {t()}s")
    return bits

def bits_to_arr(bits):
    with timer("bits_to_arr") as t:
        res = np.array([i for i, bit in enumerate(bits) if bit])
    print(f"bits_to_arr took {t()}s")
    return res