import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from mbs.utils.enums import SUBGROUPS

def main():
    df = pd.DataFrame()
    fig, axes = plt.subplots(1, 3, figsize=(18, 3))
    # report = {}
    for metric, ax in zip(['auroc', 'acc', 'f1'], axes):
        # values =
        for config in ['mbmf', 'radiomics', 'scratch']:
            report = pd.read_excel(f'MB-data/metrics/test/{config}/report.xlsx', index_col=0)
            for subgroup in SUBGROUPS + ['macro']:
                df.loc[config, subgroup] = report.loc[subgroup, metric]
            # df.loc[config, 'AVG'] = report.loc[]
        # glue = sns.load_dataset("glue").pivot(index="Model", columns="Task", values="Score")
        sns.heatmap(df, annot=True, fmt='.3f', ax=ax, cmap='crest')
        ax.set_title(metric)
    fig.savefig('heatmap.pdf')
    plt.show()

if __name__ == '__main__':
    main()
