import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def cmp(diff: np.array):
    indexes = ['WNT', 'SHH', 'G3', 'G4', 'macro', 'micro']
    columns = ['sen', 'spe', 'acc', 'auroc']
    _, p_value = wilcoxon(diff)
    return pd.DataFrame(p_value, index=indexes, columns=columns)

def main():
    m = np.load('plot/all/bs_metrics.npy')
    m_scratch = np.load('plot-scratch/all/bs_metrics.npy')
    m_rad = np.load('radiomics/plot/MLP/all/bs_metrics.npy')[:, :, 1:]  # exclude F1
    report_seg = cmp(m - m_scratch)
    report_rad = cmp(m - m_rad)
    print(report_seg)
    print(report_rad)

if __name__ == '__main__':
    main()
