# %%
from pprint import pprint

from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

import uutils
from uutils import torch_uu

save_plot: bool = True
show: bool = True
grid: bool = True

title = 'Learnig & Evaluation Curves'

tag1 = f'Train loss'
tag3 = f'Val lossrnin '

# - sinusoid
file2save = 'no_overfit'
experiment_stats = uutils.load_json('~/Desktop/paper_figs/logs_Nov23_11-26-20_jobid_438708.iam-pbs/', 'experiment_stats.json')
ckpt = torch_uu.load('~/Desktop/paper_figs/logs_Nov23_11-26-20_jobid_438708.iam-pbs/', 'ckpt_file.pt')

# - meta-overfitting
file2save = 'meta_overfit1'
experiment_stats = uutils.load_json('~/Desktop/paper_figs/logs_Nov23_11-39-21_jobid_438713.iam-pbs/', 'experiment_stats.json')
ckpt = torch_uu.load('~/Desktop/paper_figs/logs_Nov23_11-39-21_jobid_438713.iam-pbs/', 'ckpt_file.pt')

# -- get ckpt
args = ckpt['args']

episodes_train_x = np.array([args.log_train_freq * (i + 1) for i in range(len(experiment_stats['train']['loss']))])
episodes_eval_x = np.array([args.log_val_freq * (i + 1) for i in range(len(experiment_stats['eval_stats']['mean']['loss']))])


# i = 174
i = list(episodes_eval_x).index(50_00)

experiment_stats['train']['its'] = episodes_train_x[:i]
experiment_stats['eval']['its'] = episodes_eval_x[:i]

# - get figure with two axis, loss above and accuracy bellow
# fig, (loss_ax1, acc_ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
# fig, (loss_ax1, acc_ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

# - plot stuff into loss axis
plt.plot(experiment_stats['train']['its'], experiment_stats['train']['loss'][:i],
              label=tag1, linestyle='-', marker='x', color='r', linewidth=1)
plt.plot(experiment_stats['eval']['its'], experiment_stats['eval_stats']['mean']['loss'][:i],
              label=tag3, linestyle='-', marker='x', color='m', linewidth=1)

plt.legend()
plt.title(title)
plt.xlabel('(meta) epochs')
plt.ylabel('Loss')
plt.grid(grid)

plt.tight_layout()

if save_plot:
    print(f'{file2save=}')
    log_root = Path('~/Desktop').expanduser()
    plt.savefig(log_root / f'{file2save}.pdf')
    plt.savefig(log_root / f'{file2save}.svg')
    plt.savefig(log_root / f'{file2save}.png')


plt.show() if show else None

plt.close()
