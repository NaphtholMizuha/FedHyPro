from collections import namedtuple
from experiment import Experiment
import argparse
import tomli
import sqlite3
repeat = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="conf/resnet.toml")
    parser.add_argument('--nclient', type=int, default=10)
    parser.add_argument('--clip', type=float, default=10)
    parser.add_argument("--db", type=str, default="db/data.db")
    args = parser.parse_args()
    
    with open(args.config, "rb") as f:
        cfg = tomli.load(f)
    
    if args.nclient is not None:
        cfg['n_client'] = parser.parse_args().nclient
    
    if args.clip is not None:
        cfg['dp']['clip_thr'] = parser.parse_args().clip
        
    if args.db is not None:
        cfg['db'] = parser.parse_args().db
        
    if 'db' in cfg:
        conn = sqlite3.connect(cfg['db'])
        cur = conn.cursor()

    for i in range(repeat):
        experiment = Experiment(cfg)
        DB = namedtuple('DB', ['conn', 'cur'])
        experiment.execute2(i, DB(conn, cur), hp=True)
    
    if 'db' in cfg:
        conn.close()