#%%

# ml + loss vs std (best, best meta-val)

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

print('running')

# save_plot = False
save_plot = True

stds = [0.5, 1.0, 2.0, 4.0]

meta_test_loss = [7.625178372740746, 10.158726910948754, 8.668823894679546, 10.927158722639085]
meta_test_loss_std = [0.5123193061123283, 0.6722101351222379, 0.7487423250321502, 1.1042277611931715]

# these are integral difference
# meta_test_loss = [18.06054182490524, 21.024920472624352, 59.649208159349406, 53.42566085720061]
# meta_test_loss_std = [3.9390681461748627, 5.168308546143307, 4.093851180247897, 5.33135624648043]

meta_test_cca = [0.533019460241, 0.5275834341843922, 0.5326053589582443, 0.5199869662523269]
meta_test_cca_std = [0.031491566797677686, 0.019773647559281876, 0.009824007704873539, 0.015287316810794021]

meta_cca_anil_mean = [0.12, 0.12, 0.12, 0.1]

meta_test_ned = [0.4536823472286004, 0.4746091389964411, 0.4771619125271688, 0.5002327398097794]
meta_test_ned_std = [0.005222276584721787, 0.008115222132595508, 0.004904429800945309, 0.012230113038618864]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(stds, meta_test_cca, yerr=meta_test_cca_std, marker='x', label='dCCA')
axs[0].errorbar(stds, meta_test_ned, yerr=meta_test_ned_std, marker='x', label='NED')
axs[0].axhline(y=0.12, color='r', linestyle='--', label='dCCA previous work')


axs[0].legend()
axs[0].set_title('Representation difference after adaptation vs std of data set')
# axs[0].set_xlabel('std of task')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(stds, meta_test_loss, yerr=meta_test_loss_std, marker='x', label='loss', color='green')
axs[1].legend()
axs[1].set_title('Meta-Validation loss vs std of task')
axs[1].set_xlabel('std of task')
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

#%%

# ml + loss vs std (overfitted)

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

print('running')

save_plot = True

stds = [0.5, 1.0, 2.0, 4.0]

meta_test_loss = [14.435317705392839, 15.402972025811673, 20.754535799860953, 26.480839039444923]
meta_test_loss_std = [0.923371014450777, 1.4882664230227654, 2.442116138510176, 3.681431986541891]

# these are integral difference
# meta_test_loss = [18.06054182490524, 21.024920472624352, 59.649208159349406, 53.42566085720061]
# meta_test_loss_std = [3.9390681461748627, 5.168308546143307, 4.093851180247897, 5.33135624648043]

meta_test_cca = [0.49178904692331954, 0.46809875667095185, 0.486220246553421, 0.4676437427600225]
meta_test_cca_std = [0.02182605743788561, 0.028187421808105347, 0.03587469482655096, 0.03174861618592386]

meta_test_ned = [0.48837833467868935, 0.4993168077228578, 0.516462456016512, 0.4577747890933921]
meta_test_ned_std = [0.047999769721217306, 0.03752526414245839, 0.00837435717641569, 0.011480975581913778]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(stds, meta_test_cca, yerr=meta_test_cca_std, marker='x', label='dCCA')
axs[0].errorbar(stds, meta_test_ned, yerr=meta_test_ned_std, marker='x', label='NED')
axs[0].axhline(y=0.12, color='r', linestyle='--', label='dCCA previous work')

axs[0].legend()
axs[0].set_title('Representation difference after adaptation vs std of data set')
# axs[0].set_xlabel('std of task')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(stds, meta_test_loss, yerr=meta_test_loss_std, marker='o', label='loss', color='green')
axs[1].legend()
axs[1].set_title('Meta-Validation loss vs std of task')
axs[1].set_xlabel('std of task')
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


# https://seaborn.pydata.org/examples/errorband_lineplots.html


