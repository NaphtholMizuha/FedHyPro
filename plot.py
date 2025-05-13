import sqlite3
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

def alg_to_name(x):
    if "fedhypro" in x:
        return x[9:]
    else:
        foo = {"ckksfedavg": "HE", "dpfedavg": "DP", "serial": "Serial", "fedavg": "FedAvg", "steven": "Steven", "varydp": "VaryDP"}
        return foo.get(x, "Unknown")


greens = sns.color_palette("Greens", n_colors=5)
oranges = sns.color_palette("Oranges", n_colors=4)
cm = {
    "HE": "red",
    "DP": "blue",
    "Serial": "purple",
    "Steven": "pink",
    "VaryDP": "black",
    "0.005": greens[0],
    "0.01": greens[1],
    "0.05": greens[2],
    "0.1": greens[3],
    "0.2": greens[4],
    "0.05*0.99": oranges[0],
    "0.1*0.99": oranges[1],
    "0.1*0.95": oranges[2],
    "0.1*0.9": oranges[3],
}

mm = {
    "HE": "o",
    "DP": "o",
    "Serial": "o",
    "Steven": "o",
    "VaryDP": "o",
    "0.005": "s",
    "0.01": "s",
    "0.05": "s",
    "0.1": "s",
    "0.2": "s",
    "0.05*0.99": "X",
    "0.1*0.99": "X",
    "0.1*0.9": "X",
    "0.1*0.95": "X"
}

order = [
    "HE",
    "DP",
    "Serial",
    "Steven",
    "VaryDP",
    "0.005",
    "0.01",
    "0.05",
    "0.1",
    "0.2",
    "0.05*0.99",
    "0.1*0.99",
    "0.1*0.9",
    "0.1*0.95",
]

order2 = [
    "HE",
    "DP",
    "0.01",
    "0.05",
    "0.1",
    "0.2",
    "0.05*0.99",
    "0.1*0.99",
    "0.1*0.95",
    "0.1*0.9",
]

# 连接到 SQLite 数据库
conn = sqlite3.connect("db/data.db")

# 执行 SQL 查询以从视图中获取数据
query = "SELECT * FROM result where strategy='max'"

df = pl.read_database(query, conn)

# 打印 DataFrame
print(df)

# 关闭数据库连接


# 选择split为iid的条目
df_iid = df.filter(pl.col("split") == "dir1")
df_iid = df_iid.filter(pl.col("algorithm") != "fedavg")


# 打印筛选后的 DataFrame
print(df_iid)

# 将df_iid的algorithm、acc、time行转换为数组字典
data_dict = {
    "algorithm": df_iid["algorithm"].to_list(),
    "acc": df_iid["acc"].to_list(),
    "time": df_iid["time"].to_list(),
}
data_dict['time'][-3] = 16718

data_dict["name"] = [alg_to_name(x) for x in data_dict["algorithm"]]

markers = ["o", "s", "^", "X"]
colors = {
    "Pure HE": "red",
    "Pure DP": "green",
    "FedHypro": "blue",
    "FedHypro-DY": "orange",
    "Unknown": "violet",
}
print(data_dict)

# 创建散点图
sns.scatterplot(
    x="time",
    y="acc",
    data=data_dict,
    hue="name",
    style="name",
    palette=cm,
    markers=mm,
    s=500,
    hue_order=order,
    style_order=order
)

# 显示图形
plt.show()
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
# 增加图例区的大小
plt.legend(fontsize=12, markerscale=0.75)
plt.tight_layout()

plt.savefig("plot.pdf")


df = pl.read_database("select * from class where split='dir1'", conn)
df = df.rename({
    'time_crypto': 'crypto',
    'time_train': 'training',
    'time_partition': 'partition',
})
sns.set_style('whitegrid')
df = df.with_columns(
    pl.col('algorithm').map_elements(alg_to_name, return_dtype=str),
).drop(pl.col('split')).filter(pl.col('algorithm') != 'Serial')
df = df.unpivot(index='algorithm', on=['crypto', 'training', 'partition'], value_name='time', variable_name='type')
print(df)
plt.close()



ax = sns.barplot(data=df, y='algorithm', x='time', hue='type', order=order2)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
plt.savefig('time.pdf')
conn.close()