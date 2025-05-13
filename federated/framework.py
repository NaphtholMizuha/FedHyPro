from copy import deepcopy
import numpy as np
from federated.client import FedClient
from federated.server import FedServer
from typing import List
from protector.ckks import Ckks
from protector.dp import Dp
from utils.param import state_dict_to_flatten, flatten_to_state_dict, unquantize, quantize
from utils.timer import timer
from rich.console import Console
from typing import Protocol
import torch
import polars as pl
console = Console()


class Fl(Protocol):
    def run(self, n_round):
        pass


class FedAvg:
    def __init__(
        self,
        clients: List[FedClient],
        server: FedServer,
    ) -> None:
        self.clients = clients
        self.server = server

    def run_a_round(self, r):
        with timer("") as t:
            weights = []
            for client in self.clients:
                client.local_train()
                weight = client.model.state_dict()
                weight = state_dict_to_flatten(weight)
                weights.append(weight)
            weight_g = self.server.aggr_weight(weights)
            weight_g = flatten_to_state_dict(
                weight_g, self.clients[0].model.state_dict()
            )
            for client in self.clients:
                client.model.load_state_dict(weight_g)
        loss, acc = self.clients[0].test()
        return {"loss": loss, "acc": acc, "time_local_train": t()}


    def run(self, n_round):
        for r in range(n_round):
            result = self.run_a_round(r)
            print(f"Round {r+1} Loss: {result['loss']:.4f}, Acc: {result['acc']:.4f}")
            yield result


class DpFedAvg(FedAvg):
    def __init__(self, clients: List[FedClient], server: FedServer, vary=False) -> None:
        super().__init__(clients, server)
        self.vary = vary

    def run_a_round(self, r):
        with timer("") as t:
            weights = []
            for client in self.clients:
                client.local_train()
                client.calc_update()
                update = client.protector.perturb(client.update)
                update = state_dict_to_flatten(update)
                weights.append(update)
            update_g = self.server.aggr_weight(weights)
            for client in self.clients:
                client.apply_raw_update(update_g)
        loss, acc = self.clients[0].test()
        return {"loss": loss, "acc": acc, "time_local_train": t()}


class Serial(FedAvg):
    def __init__(self, clients: List[FedClient], server: FedServer) -> None:
        super().__init__(clients, server)
        for client in self.clients:
            client.gen_bn_indices()

    def run_a_round(self, r):
        with timer("") as t:
            weights = []
            for client in self.clients:
                client.local_train()
                client.calc_update()
                update = client.get_serial_update()
                weights.append(update)
            update_g = self.server.aggr_weight(weights)
            for client in self.clients:
                u = deepcopy(update_g)
                u = client.protector.decrypt(u)
                u = np.array(u)
                client.apply_raw_update(u)
        loss, acc = self.clients[0].test()
        return {"loss": loss, "acc": acc, "time_local_train": t()}


class CkksFedAvg(FedAvg):
    def __init__(self, clients: List[FedClient], server: FedServer, ckks: Ckks) -> None:
        super().__init__(clients, server)
        self.ckks = ckks

    def run_a_round(self, r):
        time_local_train, time_agg_update, time_enc, time_dec = 0, 0, 0, 0
        weights = []
        for client in self.clients:
            with timer("") as t:
                client.local_train()
            time_local_train += t()
            weight = client.model.state_dict()
            weight = state_dict_to_flatten(weight)

            with timer("") as t:
                weight = self.ckks.encrypt(weight)
            time_enc += t()

            weights.append(weight)

        with timer("") as t:
            weight_g = self.server.aggr_weight(weights)
        time_agg_update += t()

        for client in self.clients:
            w = deepcopy(weight_g)
            with timer("") as t:
                w = self.ckks.decrypt(w)
            time_dec += t()

            w = np.array(w)
            w = flatten_to_state_dict(w, self.clients[0].model.state_dict())
            client.model.load_state_dict(w)
        loss, acc = self.clients[0].test()
        return {
            "loss": loss,
            "acc": acc,
            "time_local_train": time_local_train,
            "time_agg_update": time_agg_update,
            "time_enc": time_enc,
            "time_dec": time_dec,
        }


class FedHybrid(FedAvg):
    def __init__(
        self,
        clients: List[FedClient],
        server: FedServer,
    ) -> None:
        super().__init__(clients, server)

    def run_a_round(self, r):
        masks = []
        (
            time_local,
            time_calc_mask,
            time_agg_mask,
            time_agg_update,
            time_enc,
            time_dec,
        ) = 0, 0, 0, 0, 0, 0
        for i, client in enumerate(self.clients):
            # local training
            with timer(f"Client {i} training") as t:
                client.local_train()
                client.calc_update()
            console.print(f"[yellow]Client {i} trained with time {t():.2f}[/yellow]")
            time_local += t()
            # calculate update and store in the client object
            with timer(f"Client {i} update calculation") as t:
                mask = client.get_mask()
            console.print(
                f"[yellow]Mask generated for client {i} with time {t():.2f}[/yellow]"
            )
            time_calc_mask += t()
            masks.append(mask)

        with timer("Mask aggregation") as t:
            mask_g = self.server.aggr_mask(masks)
        console.print(f"[yellow]Masks aggregated with time {t():.2f}[/yellow]")
        time_agg_mask += t()
        updates = []

        with timer("Update encryption") as t:
            for i, client in enumerate(self.clients):
                he_part, dp_part = client.get_protected_update(mask_g)
                updates.append((i, he_part, dp_part))
        console.print(f"[yellow]Update generated with time {t():.2f}[/yellow]")
        time_enc += t()

        with timer("Update aggregation") as t:
            update_g = self.server.aggr_update(updates)
        console.print(f"[yellow]Update aggregated with time: {t():.2f}[/yellow]")
        time_agg_update += t()

        with timer("Update decryption") as t:
            for client in self.clients:
                client.apply_update(update_g, mask_g)
        console.print(f"[yellow]Model distributed with time: {t():.2f}[/yellow]")
        time_dec += t()

        loss, acc = self.clients[0].test()

        result = {
            "loss": loss,
            "acc": acc,
            "time_local_train": time_local,
            "time_calc_mask": time_calc_mask,
            "time_agg_mask": time_agg_mask,
            "time_agg_update": time_agg_update,
            "time_enc": time_enc,
            "time_dec": time_dec,
        }

        return result

class Steven(FedAvg):
    def __init__(
        self,
        clients: List[FedClient],
        server: FedServer,
    ) -> None:
        super().__init__(clients, server)
            
    @staticmethod
    def get_scales(update):
        qparam = {}
        for key, value in update.items():
            min_val = value.min()
            max_val = value.max()
            scale = (max_val - min_val) / (2 ** 16 - 1) + 1e-8
            qparam[key] = (min_val, scale)
        return qparam

    @staticmethod
    def agg_secret(secrets):
        secret_sum = None
        for i in range(len(secrets)):
            if i == 0:
                secret_sum = secrets[0]
            else:
                for j in range(len(secrets)):
                    secret_sum[j] += secrets[i][j]
        return secret_sum

    def run_a_round(self, r):
        time_local, time_enc, time_agg_update, time_dec = 0, 0, 0, 0
        for i, client in enumerate(self.clients):
            # local training
            with timer(f"Client {i} training") as t:
                client.local_train()
                client.calc_update()
                client.update_dp = client.protector.perturb(client.update)
            console.print(f"[yellow]Client {i} trained with time {t():.2f}[/yellow]")
            time_local += t()
            # calculate update and store in the client object
           
        updates = []
        shares = []
        qparam = self.get_scales(self.clients[0].update_dp)
        # print(qparam)
        with timer("Update encryption") as t:
            for i, client in enumerate(self.clients):
                
                grad = quantize(client.update_dp, qparam)
                grad = client.protector.protect(grad)
                updates.append(grad)
                shares.append(client.protector.split_secret())
        console.print(f"[yellow]Update generated with time {t():.2f}[/yellow]")
        
        time_enc += t()
        update_g = None
        with timer("Update aggregation") as t:
            update_g = self.server.aggr_gf(updates)
        console.print(f"[yellow]Update aggregated with time: {t():.2f}[/yellow]")
        time_agg_update += t()
        # print(f'truth: {self.server.aggr_weight(grads)}')
        with timer("Update decryption") as t:
            shares = self.agg_secret(shares)
            for client in self.clients:
                ug_i = client.protector.recover(update_g.copy(), shares) / 10
                # print(f'pred: {ug_i}')
                ug_i = unquantize(ug_i, client.state, qparam)
                client.apply_update_statedict(ug_i)
        console.print(f"[yellow]Model distributed with time: {t():.2f}[/yellow]")
        time_dec += t()

        loss, acc = self.clients[0].test()

        result = {
            "loss": loss,
            "acc": acc,
            "time_local_train": time_local,
            "time_calc_mask": 0,
            "time_agg_mask": 0,
            "time_agg_update": time_agg_update,
            "time_enc": time_enc,
            "time_dec": time_dec,
        }

        return result
