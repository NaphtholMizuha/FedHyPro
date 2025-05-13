import polars as pl

df_main = pl.DataFrame()
for repeat in range(5):
    df = pl.read_csv(f"out/resnet18-fedavg-{repeat}.csv")
    max_loss = df['loss'].max()
    max_acc = df['acc'].max()
    total_time = df['time'].sum()
    df_sum = pl.DataFrame({
        'loss': max_loss,
        'acc': max_acc,
        'time': total_time
    })
    df_main = pl.concat([df_main, df_sum])
print(df_main)
df_mean = df_main.mean()
print(df_mean)
df_mean.write_csv("out/resnet18-summary-fedavg.csv")
    