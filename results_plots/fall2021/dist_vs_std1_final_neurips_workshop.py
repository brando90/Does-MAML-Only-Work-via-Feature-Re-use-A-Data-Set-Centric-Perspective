#%%

# ml + loss vs std (best, best meta-val)

# import matplotlib
# matplotlib.rcParams['text.usetex'] = True

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

print('running')

# save_plot = False
save_plot = True

stds = [0.5, 1.0, 2.0, 4.0]

meta_test_loss = [5.661278141538302, 3.4916918516159057, 3.8692449072996777, 5.661278141538302]
# meta_test_loss_std = [0.5123193061123283, 0.6722101351222379, 0.7487423250321502, 1.1042277611931715]

# these are integral difference
# meta_test_loss = [18.06054182490524, 21.024920472624352, 59.649208159349406, 53.42566085720061]
# meta_test_loss_std = [3.9390681461748627, 5.168308546143307, 4.093851180247897, 5.33135624648043]

meta_test_cca = [0.3233, 0.3246, 0.3016, 0.3507]
meta_test_cca_std = [0.0902, 0.0707, 0.0734, 0.0948]

meta_cca_anil_mean = [0.12, 0.12, 0.12, 0.1]


# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(stds, meta_test_cca, yerr=meta_test_cca_std, marker='x', label='dCCA')
axs[0].axhline(y=0.12, color='r', linestyle='--', label='dCCA previous work [15]')


axs[0].legend()
axs[0].set_title('Representation difference after adaptation vs std1 of data set')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

# axs[1].errorbar(stds, meta_test_loss, yerr=meta_test_loss_std, marker='x', label='loss', color='green')
axs[1].plot(stds, meta_test_loss, marker='x', label='loss', color='green')
axs[1].legend()
axs[1].set_title('Meta-Validation loss vs std1 of data set')
axs[1].set_xlabel('std1 of data set')
axs[1].set_ylabel('Loss')

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'best_relu_vs_std.png')
    plt.savefig(root / 'best_relu_vs_std.svg')
    plt.savefig(root / 'best_relu_vs_std.pdf')

plt.show()

print('done')

#%%

# #%%
#
# # ml + loss vs std (overfitted)
#
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

print('running')

save_plot = True

stds = [0.5, 1.0, 2.0, 4.0]

meta_test_loss = [6.511950352787972, 8.118141770362854, 12.123411305745442, 11.492730836073557]
# meta_test_loss_std = [0.923371014450777, 1.4882664230227654, 2.442116138510176, 3.681431986541891]

# these are integral difference
# meta_test_loss = [18.06054182490524, 21.024920472624352, 59.649208159349406, 53.42566085720061]
# meta_test_loss_std = [3.9390681461748627, 5.168308546143307, 4.093851180247897, 5.33135624648043]

meta_test_cca = [0.3073, 0.2882, 0.3267, 0.2777]
meta_test_cca_std = [0.1326, 0.1451, 0.1413, 0.1230]


fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(stds, meta_test_cca, yerr=meta_test_cca_std, marker='x', label='dCCA')
axs[0].axhline(y=0.12, color='r', linestyle='--', label='dCCA previous work [15]')

axs[0].legend()
axs[0].set_title('Representation difference after adaptation vs std1 of data set')
# axs[0].set_xlabel('std of data set')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].plot(stds, meta_test_loss, marker='x', label='loss', color='green')
axs[1].legend()
axs[1].set_title('Meta-Validation loss vs std1 of data set')
axs[1].set_xlabel('std1 of data set')
axs[1].set_ylabel('Loss')

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'relu_metaoverfitted.png')
    plt.savefig(root / 'relu_metaoverfitted.svg')
    plt.savefig(root / 'relu_metaoverfitted.pdf')

plt.show()

print('done')

#%%


