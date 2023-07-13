# importing the required library
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('GMM_BestPCA_FC.csv')
#sort by NonTarget
df = df.sort_values(by=['NonTarget', 'Target'])

Cprim = df['Cprim'].values

fig, ax = plt.subplots(figsize=(10, 4))
fig.tight_layout()
g = sns.barplot(x = 'Target', y = 'Cprim', hue = 'NonTarget', data = df, palette = 'coolwarm')
for i, bar in enumerate(g.patches):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2. -0.05, 1.05 * height, Cprim[i], rotation=45)

# max height of the figure
plt.ylim(0, 0.5)
# show legend outside the figure with NonTarget labels
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='NonTarget')

#margin
plt.subplots_adjust(right=0.8)
plt.savefig("figures_gmm/{}.svg".format("GMM_BestPCA_FC"))