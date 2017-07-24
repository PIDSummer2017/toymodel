import pandas as pd

df = pd.read_csv('pid-16599.test1e1pLE_filler_20000.csv')

muons = df.loc[df['score02'] > 0.9]

print muons.entry
