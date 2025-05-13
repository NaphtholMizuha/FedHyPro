import polars as pl
import tomli_w

df_main = pl.DataFrame()

for i in range(20):
    df = pl.read_csv(f'./temp/norm_{i}.csv')
    df_main = df_main.vstack(df)
    
print(df_main.sum())

# Convert the DataFrame to a dictionary
df_dict = df_main.mean().to_dict(as_series=False)
clip = {key: value[0] for key, value in df_dict.items()}

# Write the dictionary to a TOML file
with open('clip.toml', 'wb') as f:
    tomli_w.dump(clip, f)
