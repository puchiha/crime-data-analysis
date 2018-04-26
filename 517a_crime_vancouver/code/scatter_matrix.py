import pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
import sys

filename=sys.argv[1]
df = pd.read_csv(filename)
sns.pairplot(
    df,
    hue='CLASSIFICATION',
    vars=[c for c in df.columns if c != 'CLASSIFICATION'],
    diag_kind='kde',
    plot_kws={'s':5}
)
plt.show()
