from experiment import Experiment
import tomli
import sys

rates = [0.005, 0.01, 0.05, 0.1, 0.2]
dy_rates = [0.1, 0.2, 0.3, 0.4]
punish = [0.98, 0.97, 0.95, 0.93]
aggr = ["min", "rand", "max"]
split = ["iid", "dir0.5", "dir1"]
repeat = 5
path = "./conf"

if __name__ == "__main__":
    model = sys.argv[1]
    split = sys.argv[2]
    dynamic = sys.argv[3] == "dy"
    filename = f"{path}/{model}-{split}.toml"
    with open(filename, "rb") as f:
        conf_default = tomli.load(f)
    conf_default["split"] = split
    
    

    # if dynamic:
    #     for rate, punish in zip(dy_rates, punish):
    #         conf = conf_default.copy()
    #         conf["he_rate"] = rate
    #         conf["dynamic"]["enabled"] = True
    #         conf["dynamic"]["punish"] = punish
    #         for r in range(repeat):
    #             conf["repeat"] = r
    #             exp = Experiment(conf)
    #             exp.execute2(r)
    # else:
    #     for rate in rates:
    #         conf = conf_default.copy()
    #         conf["he_rate"] = rate
    #         for r in range(repeat):
    #             conf["repeat"] = r
    #             exp = Experiment(conf)
    #             exp.execute2(r)
