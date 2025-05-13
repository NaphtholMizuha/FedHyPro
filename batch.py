import subprocess as sp

algs = ["serial"]
for alg in algs:
    log_file = f'log/{alg}.txt'
    command = f"/home/wuzihou/miniconda3/envs/FL/bin/python run.py -a {alg} -c conf/fedhypro.toml > {log_file}"
    sp.run(["screen", "-dmS", alg, "bash", "-c", command])


# init_rates = [0.05, 0.1, 0.1, 0.1]
# punishes = [0.99, 0.99, 0.95, 0.9]
# for init_rate, punish in zip(init_rates, punishes):
#     title = f"{init_rate}x{punish}"
#     log_file = f"log/{title}.txt"
#     command = f"/home/wuzihou/miniconda3/envs/FL/bin/python run.py -a fedhypro -r {init_rate} -p {punish} -d True -c conf/fedhypro.toml > {log_file}"
#     sp.run(["screen", "-dmS", title, "bash", "-c", command])



# rates = [0.01, 0.05, 0.1, 0.2]
# for rate in rates:
#     log_file = f'log/{rate}.txt'
#     command = f"/home/wuzihou/miniconda3/envs/FL/bin/python run.py -a fedhypro -r {rate} -c 'conf/fedhypro.toml' > {log_file}"
#     sp.run(["screen", "-dmS", str(rate), "bash", "-c", command])

# strategies = ['min', 'rand']
# for strategy in strategies:
#     log_file = f'log/{strategy}.txt'
#     command = f"/home/wuzihou/miniconda3/envs/FL/bin/python run.py -s {strategy} > {log_file}"
#     screen_name = strategy
#     sp.run(["screen", "-dmS", screen_name, "bash", "-c", command])

# clips = [0.01, 0.1, 0.5]
# for clip in clips:
#     title = f"clip{clip}"
#     log_file = f'log/{title}.txt'
#     command = f"/home/wuzihou/miniconda3/envs/FL/bin/python run2.py --clip {clip} > {log_file}"
#     sp.run(["screen", "-dmS", title, "bash", "-c", command])

# n_clients = [5, 10, 25, 50]
# for n_client in n_clients:
#     title = f"nclient{n_client}"
#     log_file = f'log/{title}.txt'
#     command = f"/home/wuzihou/miniconda3/envs/FL/bin/python run2.py --nclient {n_client} > {log_file}"
#     sp.run(["screen", "-dmS", title, "bash", "-c", command])

