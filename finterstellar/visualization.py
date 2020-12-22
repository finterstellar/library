import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


ScalarFormatter().set_scientific(False)
font = 'NanumSquareRound, AppleGothic, Malgun Gothic, DejaVu Sans'
plt.style.use('bmh')
plt.rcParams['font.family'] = font
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['lines.antialiased'] = True
plt.rcParams['figure.figsize'] = [10.0, 5.0]
plt.rcParams['savefig.dpi'] = 96
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'


def str_to_list(s):
    if type(s) == list:
        cds = s
    else:
        cds = []
        cds.append(s)
    return cds


def draw_chart(df, left=None, right=None, log=False):
    fig, ax1 = plt.subplots()
    x = df.index
    if left is not None:
        left = str_to_list(left)
        i = 1
        for c in left:
            ax1.plot(x, df[c], label=c, color='C'+str(i), alpha=1)
            i += 1
        if log:
            ax1.set_yscale('log')
            ax1.yaxis.set_major_formatter(ScalarFormatter())
            ax1.yaxis.set_minor_formatter(ScalarFormatter())
    # secondary y
    if right is not None:
        right = str_to_list(right)
        ax2 = ax1.twinx()
        i = 6
        for c in right:
            ax2.plot(x, df[c], label=c+'(R)', color='C'+str(i), alpha=0.7)
            ax1.plot(np.nan, label=c+'(R)', color='C'+str(i))
            i += 1
        ax2.grid(False)
        if log:
            ax2.set_yscale('log')
            ax2.yaxis.set_major_formatter(ScalarFormatter())
            ax2.yaxis.set_minor_formatter(ScalarFormatter())
    ax1.legend(loc=2)