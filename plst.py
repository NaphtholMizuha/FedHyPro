import sqlite3
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
conn = sqlite3.connect("db/data.db")

query = "SELECT strategy, round, accuracy * 100 as acc FROM trains where algorithm='fedhypro-0.1'"
df = pl.read_database(query, conn).sort('strategy')
sns.set_style("whitegrid")
sns.set_palette("Set2")
sns.lineplot(data=df, x="round", y="acc", hue="strategy", errorbar='sd')
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.legend(title="Strategy")
plt.xticks(np.arange(0, 20, 1))
plt.yticks(np.arange(0, 75, 5))
plt.savefig('strategy.pdf')
print(df)