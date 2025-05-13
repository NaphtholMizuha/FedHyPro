import numpy as np
from bitarray import bitarray
from typing import List


class FedServer:
    def __init__(self, datasizes) -> None:
        self.datasizes = datasizes

    def all_datasize(self) -> int:
        return sum(self.datasizes)
    
    def aggr_gf(self, weights):
        weight_g = None
        
        for i, w in enumerate(weights):
            if weight_g is None:
                weight_g = w
            else:
                weight_g += w
        return weight_g


    def aggr_update(self, updates):
        he_part, dp_part = None, None

        for client_id, he, dp in updates:
            # print(f"client_id: {client_id}, he: {he}, dp: {dp}")
            coeff = self.datasizes[client_id] / self.all_datasize()
            if he_part is None:
                he_part = he * coeff
                dp_part = dp * coeff
            else:
                he_part += he * coeff
                dp_part += dp * coeff
            # print(f"now he_part: {np.array(he_part.decrypt())}")
        return he_part, dp_part

    def aggr_weight(self, weights):
        weight_g = None
        
        for i, w in enumerate(weights):
            coeff = self.datasizes[i] / self.all_datasize()
            if weight_g is None:
                weight_g = w * coeff
            else:
                weight_g += w * coeff
        return weight_g

    def aggr_mask(self, masks: List[bitarray]) -> bitarray:
        if not masks:
            return bitarray()

        # Convert bitarrays to a numpy array for efficient computation
        bitarray_length = len(masks[0])

        # Create a 2D numpy array from the bitarrays
        bitarray_matrix = np.array([np.array(b.tolist()) for b in masks])

        # Count occurrences of 1 in each column
        column_sums = np.sum(bitarray_matrix, axis=0)
        # Global HE part's size
        he_nums = np.average(np.sum(bitarray_matrix, axis=1)).astype(int)

        # Select indices of the top he_nums columns with the highest sums
        indices = np.argpartition(-column_sums, he_nums)[:he_nums]

        # Create a new bitarray with the same number of ones
        result = bitarray(bitarray_length)
        result.setall(0)

        # Set the top positions to 1
        for i in range(he_nums):
            result[indices[i]] = 1

        return result
