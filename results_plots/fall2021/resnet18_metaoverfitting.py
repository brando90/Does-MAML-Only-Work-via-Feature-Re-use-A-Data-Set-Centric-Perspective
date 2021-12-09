#%%

# resnet18 meta-overfitting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import MaxNLocator
import uutils

from pathlib import Path

print('running')

save_plot = True
# save_plot = False


# d: list[tuple[float, int, float]] = uutils.load_json_list(path='~/Desktop/', filename='run-.-tag-meta-train_loss.json')
# its, train_losses = uutils.torch_uu.tensorboard.tensorboard_run_list_2_matplotlib_list(d)

# d: list[tuple[float, int, float]] = uutils.load_json_list(path='~/Desktop/paper_figs', filename='run-.-tag-meta-val_loss.json')
# its, val_losses = uutils.torch_uu.tensorboard.tensorboard_run_list_2_matplotlib_list(d, 0.8)

# plt.plot(its, val_losses)
# plt.show()


# Plot the responses for different events and regions
# sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=fmri)


plt.show()

# - create plot
# fig, (loss_ax1, acc_ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
#
# # - plot stuff into loss axis
# loss_ax1.plot(experiment_stats['train']['its'], experiment_stats['train']['loss'],
#       label=tag1, linestyle='-', marker='o', color='r', linewidth=1)
# loss_ax1.plot(experiment_stats['val']['its'], experiment_stats['val']['loss'],
#       label=tag3, linestyle='-', marker='o', color='m', linewidth=1)
#
# loss_ax1.legend()
# loss_ax1.set_title(title)
# loss_ax1.set_ylabel('Loss')
# loss_ax1.grid(grid)
#
# # - plot stuff into acc axis
# acc_ax2.plot(experiment_stats['train']['its'], experiment_stats['train']['acc'],
#       label=tag2, linestyle='-', marker='o', color='b', linewidth=1)
# acc_ax2.plot(experiment_stats['val']['its'], experiment_stats['val']['acc'],
#       label=tag4, linestyle='-', marker='o', color='c', linewidth=1)
#
# acc_ax2.legend()
# x_axis_label: str = args.training_mode  # epochs or iterations
# acc_ax2.set_xlabel(x_axis_label)
# acc_ax2.set_ylabel(ylabel_acc)
# acc_ax2.grid(grid)
#
# plt.tight_layout()
#
#
# if save_plot:
#     root = Path('~/Desktop').expanduser()
#     plt.savefig(root / 'pytorch_resnet18_metaoverfitting.png')
#     plt.savefig(root / 'pytorch_resnet18_metaoverfitting.svg')
#     plt.savefig(root / 'pytorch_resnet18_metaoverfitting.pdf')
#
# plt.show()
#
print('done')