from torch.utils.data import DataLoader
from federated.client import DynamicRatioClient, ProposedClient
from federated.framework import CkksFedAvg, DpFedAvg, FedAvg, FedHybrid, Serial, Steven
from federated.server import FedServer
from local.dataset import fetch_dataset
from local.datasplitter import DataSplitter
from local.model import fetch_model
from local.trainer import Trainer
from protector.core import (
    HybridProtector,
    VaryingDpProtector,
    StevenProtector,
    LayerDpProtector,
)
from protector.dp import Dp
from protector.ckks import Ckks


class Experiment:
    def __init__(self, cfg):
        self.data_name = cfg["data_name"]
        self.model_name = cfg["model_name"]
        self.n_client = cfg["n_client"]
        self.n_round = cfg["n_round"]
        self.n_epoch = cfg["n_epoch"]
        self.eps = cfg["dp"].get("eps")
        self.delta = cfg["dp"]["delta"]
        self.clip_thr = cfg["dp"]["clip_thr"]
        self.decay = cfg["dp"]["decay"]
        self.device = cfg["device"]
        self.split = cfg["split"]
        self.lr = cfg["lr"]
        self.he_rate = cfg["he_rate"]
        self.dynamic_enabled = cfg["dynamic"]["enabled"]
        self.punish = cfg["dynamic"].get("punish")
        self.aggr = cfg.get("strategy", "max")
        self.algorithm = cfg.get("algorithm")
        self.save_path = cfg.get("save_path")
        self.cfg = cfg
        print(cfg)

    def execute2(self, repeat, db, hp=False):
        if self.dynamic_enabled:
            ratio_name = f"{self.he_rate}*{self.punish}"
        else:
            ratio_name = self.he_rate

        if self.algorithm == "fedhypro":
            algorithm_name = "fedhypro_" + self.aggr
        else:
            algorithm_name = self.algorithm

        exp_name = (
            f"{self.model_name}-{ratio_name}-{algorithm_name}-{self.split}-{repeat}"
        )
        print(f"Running experiment {exp_name}")

        train_dataset, test_dataset = fetch_dataset(self.data_name)
        print("Data fetched")

        init_state = fetch_model(self.model_name, num_classes=10).state_dict()
        models = [
            fetch_model(self.model_name, num_classes=10).to(self.device)
            for _ in range(self.n_client)
        ]
        for model in models:
            model.load_state_dict(init_state)
        print("Model fetched")

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        ds = DataSplitter(
            train_dataset,
            n_client=self.n_client,
            batch_size=32,
            num_workers=4,
            path=f"./temp/{self.data_name}-c{self.n_client}-{self.split}.npy",
            method=self.split,
        )
        print("Data splitted")

        train_loaders = ds.dataloaders
        d_min = min(ds.d_sizes)

        protector = None

        if self.algorithm == "varydp":
            protector = [
                VaryingDpProtector(
                    eps=self.eps,
                    delta=self.delta,
                    d_min=d_min,
                    thr=self.clip_thr,
                    n_round=self.n_round,
                    decay=self.decay,
                )
                for _ in range(self.n_client)
            ]
        elif self.algorithm == "steven":
            protector = [
                StevenProtector(
                    n_client=self.n_client,
                    s_dim=500,
                    packet_dim=1000,
                    sigma=4,
                    eps=self.eps,
                    delta=self.delta,
                    d_min=d_min,
                    thr=self.clip_thr,
                    n_round=self.n_round,
                )
                for _ in range(self.n_client)
            ]
        elif self.algorithm == "dpfedavg" or self.algorithm == "serial":
            protector = [
                LayerDpProtector(
                    eps=self.eps,
                    delta=self.delta,
                    d_min=d_min,
                    thr=self.clip_thr,
                    n_round=self.n_round,
                )
                for _ in range(self.n_client)
            ]
        else:
            protector = [
                HybridProtector(
                    eps=self.eps,
                    delta=self.delta,
                    d_min=d_min,
                    thr=self.clip_thr,
                    n_round=self.n_round,
                )
                for _ in range(self.n_client)
            ]

        clients = None
        if self.dynamic_enabled:
            clients = [
                DynamicRatioClient(
                    model=models[i],
                    trainer=Trainer(
                        models[i],
                        train_loaders[i],
                        test_loader,
                        lr=self.lr,
                        device=self.device,
                    ),
                    protector=protector[0],
                    n_epoch=self.n_epoch,
                    he_rate=self.he_rate,
                    mask_type=self.aggr,
                    punish=self.punish,
                )
                for i in range(self.n_client)
            ]
        else:
            clients = [
                ProposedClient(
                    model=models[i],
                    trainer=Trainer(
                        models[i],
                        train_loaders[i],
                        test_loader,
                        lr=self.lr,
                        device=self.device,
                    ),
                    protector=(
                        protector[i] if self.algorithm != "fedhypro" else protector[0]
                    ),
                    n_epoch=self.n_epoch,
                    he_rate=self.he_rate,
                    mask_type=self.aggr,
                )
                for i in range(self.n_client)
            ]

        server = FedServer(ds.d_sizes)

        if self.algorithm == "fedhypro":
            print("start fedhypro")
            framework = FedHybrid(clients, server)
        elif self.algorithm == "ckksfedavg":
            print("ckksfedavg")
            ckks = Ckks()
            framework = CkksFedAvg(clients, server, ckks)
        elif self.algorithm == "dpfedavg" or self.algorithm == "varydp":
            print(self.algorithm)
            framework = DpFedAvg(clients, server)
        elif self.algorithm == "fedavg":
            framework = FedAvg(clients, server)
        elif self.algorithm == "serial":
            framework = Serial(clients, server)
        elif self.algorithm == "steven":
            framework = Steven(clients, server)
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")

        print("Start Training")
        for rnd, result in enumerate(framework.run(self.n_round)):
            # Insert the result dictionary into the 'trains' table in the SQLite database
            if hp:
                cols = [
                    "model",
                    "clip_thr",
                    "n_client",
                    "split",
                    "ordinal",
                    "round",
                    "accuracy",
                ]
                cols_str = ", ".join(cols)
                placeholders = ", ".join(["?"] * len(cols))
                sql = f"INSERT OR REPLACE INTO hp ({cols_str}) VALUES ({placeholders})"
                db.cur.execute(
                    sql,
                    (
                        self.model_name,
                        self.clip_thr,
                        self.n_client,
                        self.split,
                        repeat,
                        rnd,
                        result.get("acc", 0),
                    ),
                )
            else:
                cols = [
                    "model",
                    "algorithm",
                    "strategy",
                    "split",
                    "ordinal",
                    "round",
                    "accuracy",
                    "time_local_train",
                    "time_calc_mask",
                    "time_agg_mask",
                    "time_agg_update",
                    "time_enc",
                    "time_dec",
                ]
                cols_str = ", ".join(cols)
                placeholders = ", ".join(["?"] * 13)
                sql = f"INSERT OR REPLACE INTO trains ({cols_str}) VALUES ({placeholders})"
                alg = (
                    self.algorithm
                    if self.algorithm != "fedhypro"
                    else (
                        f"fedhypro-{self.he_rate}*{self.punish}"
                        if self.dynamic_enabled
                        else f"fedhypro-{self.he_rate}"
                    )
                )
                db.cur.execute(
                    sql,
                    (
                        self.model_name,
                        alg,
                        self.aggr,
                        self.split,
                        repeat,
                        rnd,
                        result.get("acc", 0),
                        result.get("time_local_train", 0),
                        result.get("time_calc_mask", 0),
                        result.get("time_agg_mask", 0),
                        result.get("time_agg_update", 0),
                        result.get("time_enc", 0),
                        result.get("time_dec", 0),
                    ),
                )
            db.conn.commit()
