# %%

# ml + loss vs inner steps (Sigmoid best val)

import numpy as np
import matplotlib.pyplot as plt
from pylab import MaxNLocator

from pathlib import Path

print('running')

save_plot = True
# save_plot = False

# - data for distance
inner_steps_for_dist = [1, 2, 4, 8, 16, 32]

meta_test_cca = [0.2801, 0.2866, 0.2850, 0.2848, 0.2826, 0.2914]
meta_test_cca_std = [0.0351, 0.0336, 0.0322, 0.0341, 0.0321, 0.0390]

# - data for meta-lost
inner_steps_for_loss = [0, 1, 2, 4, 8, 16, 32]

loss_maml0 = 43.43485323588053

meta_test_loss = [loss_maml0, 10.404328906536103, 4.988216777642568, 5.07447034517924, 5.449032692114512, 5.36303452650706, 4.339294484257698]

# - create plot
fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(inner_steps_for_dist, meta_test_cca, yerr=meta_test_cca_std, marker='x', label='dCCA')
# axs[0].errorbar(inner_steps_for_dist, meta_test_ned, yerr=meta_test_ned_std, marker='x', label='NED')
axs[0].axhline(y=0.12, color='r', linestyle='--', label='dCCA previous work [15]')
axs[0].legend()
axs[0].set_title('Representation difference vs adaption\'s inner steps ')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].plot(inner_steps_for_loss, meta_test_loss, marker='x', label='loss', color='g')
axs[1].set_title('Meta-Validation loss vs adaptation\'s inner steps')
axs[1].set_xlabel('adaptation\'s inner steps')
axs[1].set_ylabel('Loss')
# axs[1].axhline(y=loss_maml0, color='g', linestyle='--', label='not adaptated')
axs[1].get_xaxis().set_major_locator(MaxNLocator(integer=True))

axs[1].legend()

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'ml_loss_vs_inner_steps_sigmoid_best.png')
    plt.savefig(root / 'ml_loss_vs_inner_steps_sigmoid_best.svg')
    plt.savefig(root / 'ml_loss_vs_inner_steps_sigmoid_best.pdf')

plt.show()

#%%

# ml + loss vs inner steps (ReLU best net)

import numpy as np
import matplotlib.pyplot as plt
from pylab import MaxNLocator

from pathlib import Path

print('running')

save_plot = True
# save_plot = False

# - data for distance
inner_steps_for_dist = [1, 2, 4, 8, 16, 32]

meta_test_cca = [0.2876, 0.2962, 0.2897, 0.3086, 0.2951, 0.3024]
meta_test_cca_std = [0.0585, 0.0649, 0.0575, 0.0625, 0.0565, 0.0620]

# - data for meta-loss
inner_steps_for_loss = [0, 1, 2, 4, 8, 16, 32]

loss_maml0 = 19.27044554154078
# loss_maml0_std = 1.019144981585053

meta_test_loss = [loss_maml0,
                  5.545517734686533, 7.434794012705485, 6.754467636346817, 6.577781716982524, 3.731084116299947, 6.21407161851724]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(inner_steps_for_dist, meta_test_cca, yerr=meta_test_cca_std, marker='x', label='dCCA')
axs[0].axhline(y=0.12, color='r', linestyle='--', label='dCCA previous work [15]')
axs[0].legend()
axs[0].set_title('Representation difference vs adaption\'s inner steps ')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].plot(inner_steps_for_loss, meta_test_loss, marker='x', label='loss', color='g')
axs[1].set_title('Meta-Validation loss vs adaptation\'s inner steps')
axs[1].set_xlabel('adaptation\'s inner steps')
axs[1].set_ylabel('Loss')
# axs[1].axhline(y=loss_maml0, color='g', linestyle='--', label='not adaptated')
axs[1].get_xaxis().set_major_locator(MaxNLocator(integer=True))

axs[1].legend()

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'ml_loss_vs_inner_steps_relu_best.png')
    plt.savefig(root / 'ml_loss_vs_inner_steps_relu_best.svg')
    plt.savefig(root / 'ml_loss_vs_inner_steps_relu_best.pdf')

plt.show()

print('done')