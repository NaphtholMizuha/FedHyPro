import polars as pl
models = ["resnet18"]
names = ["0.1%", "0.5%", "1%", "5%", "10%", "20%", "dy", "0.3x0.95", "0.1x0.95", "0.2x0.97", "0.3x0.9", "0.4x0.935"]

for model in models:
    df_main = pl.DataFrame()
    accs = []
    times = []
    avg_max_values_dict = {}
    for name in names:
        max_values_list = []
        for repeat in range(5):
            df = pl.read_csv(f"out/{model}-{name}-{repeat}.csv")
            # 如果存在eps列，则删除
            if "eps" in df.columns:
                df = df.drop("eps")
            max_values = {col: df[col].max() for col in df.columns}
            max_values_list.append(max_values)
        avg_max_values = {col: sum(max_values[col] for max_values in max_values_list) / len(max_values_list) for col in max_values_list[0].keys()}
        print(avg_max_values)
        df_row = pl.DataFrame(avg_max_values)
        df_row = df_row.with_columns(pl.lit(name).alias("name")).select("name", *df_row.columns)
        total_time = sum(df[col].sum() for col in df.columns if "time" in col)
        df_row = df_row.with_columns(pl.lit(total_time).alias("total_time"))
        df_main = pl.concat([df_main, df_row], how="vertical")
    
    df_main = df_main.with_columns(
        (pl.col("acc") * 100).round(2).alias("acc"),
        *[pl.col(col).round(1).alias(col) for col in df_main.columns if "time" in col]
    ).select([col for col in df_main.columns if col != "loss"])
    df_main = df_main.select(
        pl.col("name"),
        pl.col("acc"),
        pl.col("total_time")
    )
    print(df_main)
    df_main.write_csv(f"out/{model}-summary.csv")

            