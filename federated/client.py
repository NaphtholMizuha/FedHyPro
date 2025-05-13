from abc import ABC, abstractmethod

import bitarray
import numpy as np
import torch.nn as nn

from local.trainer import Trainer
from protector.core import Protector
from utils.param import state_dict_to_flatten, flatten_to_state_dict, quantize, unquantize

from utils.timer import timer


class FedClient(ABC):
    @abstractmethod
    def local_train(self):
        """Local training"""
        pass

    @abstractmethod
    def calc_update(self) -> dict:
        """Calculate update"""
        pass

    @abstractmethod
    def get_mask(self):
        """Get mask"""
        pass

    @abstractmethod
    def get_protected_update(self, mask):
        """Get protected update
        :param mask: Mask used for update
        """
        pass

    @abstractmethod
    def apply_update(self, update, mask):
        """Apply update to model
        :param update: Stored local update
        :param mask: Mask for update
        :return: None
        """
        pass

    @abstractmethod
    def test(self):
        """Test model"""
        pass


class ProposedClient(FedClient):
    def run(self, queue, queue_update, signal):
        """Run client
        :param queue: Update queue
        :param queue_update: Queue for updates
        :param signal: Signal
        """
        pass

    count = 0

    def __init__(
        self,
        model: nn.Module,
        trainer: Trainer,
        protector: Protector,
        n_epoch: int,
        he_rate: float,
        mask_type: str = "max",
    ) -> None:
        """Initialize ProposedClient
        :param model: Model to train
        :param trainer: Trainer
        :param protector: Protector
        :param n_epoch: Number of training epochs
        :param he_rate: Proportion of homomorphic encryption part
        :param mask_type: Type of mask, "max" or "min"
        """
        self.id = ProposedClient.count
        ProposedClient.count += 1
        self.model = model
        self.trainer = trainer
        self.protector = protector
        self.n_epoch = n_epoch
        self.he_rate = he_rate
        self.state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        self.mask_type = mask_type
    def test(self):
        """Test model and return results"""
        return self.trainer.test()
    
    def gen_bn_indices(self) -> np.ndarray:
        """Generate batch normalization indices"""
        cur_idx = 0
        bn_indices = np.zeros(sum(param.numel() for param in self.state.values()), dtype=bool)
        for name, param in self.state.items():
            param_flat = param.view(-1).cpu().numpy()

            if "bn" in name:
                # Set the indices of BN layers to True
                bn_indices[cur_idx:cur_idx + len(param_flat)] = True

            cur_idx += len(param_flat)
        self.bn_indices = bn_indices

    def local_train(self):
        """Perform local training and get update"""
        self.trainer.train_epochs(self.n_epoch)  # train the model and get update

    def calc_update(self):
        """Calculate the update between current model and previous state"""
        cur = self.model.state_dict()
        self.update = {k: cur[k].cpu() - self.state[k] for k in cur.keys()}
        # return self.update

    def get_mask(self):
        """Calculate and return the update mask"""
        mask = self.calc_mask(self.update, self.he_rate)  # calculate the mask
        return mask
    
    def get_update(self):
        return state_dict_to_flatten(self.update)
    
    def get_quantized_update(self):
        return quantize(self.update)
    
    def get_perturbed_update(self):
        update = state_dict_to_flatten(self.update)
        # Filter out BN layers parameters
        update_bn, update_non_bn = update[self.bn_indices], update[~self.bn_indices]
        update_non_bn = self.protector.perturb(update_non_bn)
        update = np.empty_like(update)
        update[self.bn_indices] = update_bn
        update[~self.bn_indices] = update_non_bn
        return update
    
    def get_serial_update(self):
        return self.protector.encrypt(self.get_perturbed_update())

    def get_protected_update(self, mask):
        """Get protected update
        :param mask: Mask used for update
        :return: Protected update
        """
        update = state_dict_to_flatten(self.update)
        return self.protector.protect(update, mask)
    
    def apply_update_statedict(self, update):
        self.state = {k: self.state[k] + update[k] for k in self.state.keys()}
        self.model.load_state_dict(self.state)

    def apply_raw_update(self, update):
        """
        Apply raw update to model
        :param update: Raw update
        :return: None
        """
        update = flatten_to_state_dict(update, self.state)
        self.state = {k: self.state[k] + update[k] for k in self.state.keys()}
        self.model.load_state_dict(self.state)

    def apply_quantized_update(self, update):
        update = update.astype(np.int32)
        update = unquantize(update, self.state)
        self.state = {k: self.state[k] + update[k] for k in self.state.keys()}
        self.model.load_state_dict(self.state)

    def apply_update(self, update, mask):
        """
        Apply update to model
        :param update: Stored local update
        :param mask: Mask for update
        :return: None
        """
        he_part, dp_part = update
        update = self.protector.recover(he_part, dp_part, mask)
        update = flatten_to_state_dict(update, self.state)
        self.state = {k: self.state[k] + update[k] for k in self.state.keys()}
        self.model.load_state_dict(self.state)

    def calc_mask(self, update: dict, he_rate: float) -> bitarray.bitarray:
        """
        Calculate the update mask, indicating the top r% elements
        :param update: Stored local update
        :param he_rate: Proportion of homomorphic encryption part
        :return: Update mask
        """
        flattened = np.array([], dtype=np.float32)
        bn_indices = np.array([], dtype=np.int32)
        cur_idx = 0


        for name, param in update.items():
            param_flat = param.view(-1).cpu().numpy()
            flattened = np.concatenate((flattened, param_flat))

            if "bn" in name:
                # Save the indices of BN layers
                bn_indices = np.concatenate((bn_indices, np.arange(cur_idx, cur_idx + len(param_flat))))

            cur_idx += len(param_flat)
        flattened = np.array(flattened)
        # calculate the number of non-BN elements in the HE part
        find_num = int(len(flattened) * he_rate) - len(bn_indices)

        all_indices = np.arange(len(flattened))
        non_bn_indices = np.setdiff1d(all_indices, bn_indices)
        non_bn_params = flattened[non_bn_indices]
        non_bn_params_abs = np.abs(non_bn_params)


        if self.mask_type == "max":
            # get the indices of the top-r% elements
            top_k_indices = np.argpartition(non_bn_params_abs, -find_num)[-find_num:]
        elif self.mask_type == "rand":
            top_k_indices = np.random.choice(len(non_bn_params_abs), find_num, replace=False)
        elif self.mask_type == "min":
            # get the indices of the bottom-r% elements
            top_k_indices = np.argpartition(non_bn_params_abs, find_num)[:find_num]
        else:
            raise ValueError(f"Mask type {self.mask_type} not supported")


        top_k_indices = np.array(non_bn_indices)[top_k_indices]
        # combine the indices of BN layers and the top-r% elements
        total_indices = np.union1d(top_k_indices, bn_indices)

        mask = bitarray.bitarray(len(flattened))
        for i in total_indices:
            mask[i] = True

        return mask

class DynamicRatioClient(ProposedClient):
    def __init__(
        self,
        model: nn.Module,
        trainer: Trainer,
        protector: Protector,
        n_epoch: int,
        he_rate: float,
        punish: float,
        mask_type: str,
    ) -> None:
        super().__init__(model, trainer, protector, n_epoch, he_rate, mask_type)
        self.punish = punish

    def apply_update(self, update, mask):
        super().apply_update(update, mask)
        self.he_rate *= self.punish
        print(f"Modi he_rate: {self.he_rate}")
