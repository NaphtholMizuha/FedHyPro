from collections import namedtuple
from experiment import Experiment
import argparse
import tomli
import sqlite3

splits = ['iid', 'dir1', 'dir0.5']
repeat = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="conf/resnet.toml")
    parser.add_argument("-a", "--algorithm", type=str, default="fedhypro")
    parser.add_argument("--db", "--database", type=str, default="db/data.db")
    parser.add_argument("-d", "--dynamic", type=bool, default=None)
    parser.add_argument("-r", "--rate", type=float, default=None)
    parser.add_argument("-p", "--punish", type=float, default=None)
    parser.add_argument('-s', '--strategy', type=str, default='max')
    parser.add_argument('--clip', type=float, default=10)
    args = parser.parse_args()
    
    with open(args.config, "rb") as f:
        cfg = tomli.load(f)
    
    if args.db is not None:
        cfg['db'] = parser.parse_args().db
        
    if args.algorithm is not None:
        cfg['algorithm'] = parser.parse_args().algorithm
        
    print(args.dynamic)
    if args.dynamic is not None:
        cfg['dynamic']['enabled'] = args.dynamic
        
    if args.rate is not None:
        cfg['he_rate'] = args.rate
        
    if args.punish is not None:
        cfg['dynamic']['punish'] = args.punish
        
    cfg['strategy'] = args.strategy
        
    if 'db' in cfg:
        conn = sqlite3.connect(cfg['db'])
        cur = conn.cursor()

        
    for split in splits:
        cfg['split'] = split
        for i in range(repeat):
            experiment = Experiment(cfg)
            DB = namedtuple('DB', ['conn', 'cur'])
            experiment.execute2(i, DB(conn, cur))
    
    if 'db' in cfg:
        conn.close()