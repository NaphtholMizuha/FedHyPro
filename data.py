import sqlite3
import polars as pl

def df_to_latex(df, keys=[]):
    for row in df.iter_rows():
        print(f"{row[0]:.0f} & {row[1]} \\\\")

pl.Config.set_tbl_cols(15)
pl.Config.set_tbl_rows(-1)
# 连接到 SQLite3 数据库
conn = sqlite3.connect("db/data.db")

# 使用 Polars 从 SQLite3 数据库中读取数据
df = pl.read_database("SELECT algorithm, split, acc, time FROM result WHERE strategy='max'", conn)
df = df.with_columns((pl.col("acc") / pl.col("time") *100).alias("er").round(2))
df = df.group_by('algorithm').agg(['acc', 'time', 'er']).sort('algorithm')

for row in df.iter_rows():
    acc = [str(x) for x in row[1]]
    time = [f"{x:.0f}" for x in row[2]]
    er = [str(x) for x in row[3]]
    acc = ' & '.join(acc)
    time = ' & '.join(time)
    er = ' & '.join(er)
    print(f"{row[0]}: {acc} & {time} & {er} \\\\")

df = pl.read_database("SELECT strategy, acc, time FROM result WHERE algorithm='fedhypro-0.1' and split='dir1'", conn)
df = df.group_by('strategy').agg(['acc', 'time']).sort('strategy')
print(df)

for row in df.iter_rows():
    acc = [str(x) for x in row[1]]
    time = [f"{x:.0f}" for x in row[2]]
    acc = ' & '.join(acc)
    time = ' & '.join(time)
    print(f"{row[0]}: {acc} & {time} \\\\")
# df = df.with_columns(
#     (pl.col("time_agg_update") + pl.col("time_enc") + pl.col("time_dec")).alias(
#         "time_crypt"
#     ),
#     (
#         pl.col("time_local_train") + pl.col("time_calc_mask") + pl.col("time_agg_mask")
#     ).alias("time_common"),
# )

# df = df.with_columns((pl.col("time_crypt") + pl.col("time_common")).alias("total_time"))


# # 选择主键列、accuracy 和 total_time
# result = df.select(
#     [
#         "algorithm",
#         "strategy",
#         "split",
#         "ordinal",
#         "round",
#         "accuracy",
#         "total_time",
#     ]
# )

# result = result.sort(
#     [
#         "algorithm",
#         "strategy",
#         "split",
#         "ordinal",
#         "round",
#     ]
# )

# # 按主键分组，计算accuracy的最大值和total_time的总值
# result = result.group_by(
#     ["algorithm", "strategy", "split", "ordinal"]
# ).agg(
#     [
#         pl.col("accuracy").max().alias("max_accuracy"),
#         pl.col("total_time").sum().alias("total_time_sum"),
#     ]
# )

# result = result.group_by(
#     ["algorithm", "strategy", "split"]
# ).agg(
#     [
#         pl.col('max_accuracy').mean().alias('acc'),
#         pl.col('total_time_sum').mean().alias('time')
#     ]
# )

# result = result.drop('strategy')

# result = result.sort(
#     [
#         "split", "algorithm"
#     ],
#     descending=[True, False
                
#                 ]
# )

# # 打印 DataFrame
# result = result.with_columns(
#     (pl.col("acc") * 100).round(2).alias("acc"),
#     pl.col("time").round().alias("time")
# )

# print(result)

# result = result.group_by('algorithm').agg(['acc', 'time'])


df = pl.read_database("select * from clip_thr", conn)
df_to_latex(df)
df = pl.read_database("select * from n_client", conn)
df_to_latex(df)
# 关闭数据库连接
conn.close()
