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

meta_test_cca = [0.5977237413326899, 0.5928615580002466, 0.5958830783764522, 0.598208557566007, 0.5973852763573329, 0.5957469274600347]
meta_test_cca_std = [0.00707934713573587, 0.00846237404064218, 0.007949951903377304, 0.0065036558630005474, 0.004912136217673985, 0.00909417287489179]

meta_test_ned = [0.5173536032550894, 0.5093832578420319, 0.511174541809931, 0.5154659108732735, 0.51203569236589, 0.5070959242477404]
meta_test_ned_std = [0.016695623433091045, 0.007882302117803776, 0.020698135909910757, 0.013762821990194829, 0.02757315414211051, 0.008426892297632705]

# - data for meta-lost
inner_steps_for_loss = [0, 1, 2, 4, 8, 16, 32]

loss_maml0 = 43.96553179168701
loss_maml0_std = 1.8058700680403383

meta_test_loss = [loss_maml0, 13.739946127653122, 8.793661404013632, 8.447031805932522, 8.42278252196312, 8.477399283349515, 8.312923258244993]
meta_test_loss_std = [loss_maml0_std, 1.0994678776141051, 0.36911269154459236, 0.432820357268575, 0.7749138218896927, 0.9523679944987157, 0.9205561753211284]

# - create plot
fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(inner_steps_for_dist, meta_test_cca, yerr=meta_test_cca_std, marker='x', label='dCCA')
axs[0].errorbar(inner_steps_for_dist, meta_test_ned, yerr=meta_test_ned_std, marker='x', label='NED')
axs[0].axhline(y=0.12, color='r', linestyle='--', label='dCCA previous work')
axs[0].legend()
axs[0].set_title('Representation difference vs adaption\'s inner steps ')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(inner_steps_for_loss, meta_test_loss, yerr=meta_test_loss_std, marker='x', label='loss', color='g')
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
inner_steps_for_dist = [1, 2, 3, 4, 5, 8, 16, 32]

meta_test_cca = [0.5554760853449504, 0.5614992250998815, 0.5559805711110433, 0.5665635496377945, 0.5695259730021158, 0.5757676988840104, 0.5487342963616053, 0.5536342511574428]
meta_test_cca_std = [0.024697060891281774, 0.01635608540630873, 0.020506618487168466, 0.021074992715715904, 0.008751912905302727, 0.013803703536447796, 0.0040189992442360494, 0.01672345811738925]

meta_test_ned = [0.509952854503624, 0.49967966980659717, 0.5071488178037521, 0.509229507290115, 0.5086124367750696, 0.5105350886485593, 0.507246109916044, 0.5101748466011903]
meta_test_ned_std = [0.009346849133655227, 0.015704240939143948, 0.0024862043152066556, 0.009187025418748784, 0.016039790497987778, 0.0038458444722408187, 0.005801110523967555, 0.007980516656024433]
assert len(inner_steps_for_dist) == len(meta_test_ned)

# - data for meta-loss
inner_steps_for_loss = [0, 1, 2, 3, 4, 5, 8, 16, 32]

loss_maml0 = 19.455371450126172
loss_maml0_std = 1.019144981585053

meta_test_loss = [loss_maml0, 8.909195451736451, 10.453863927662372, 10.438749479651452, 10.866869341522456, 10.897315060257913, 8.694210491478442, 9.191108769118786, 9.092517228752374]
meta_test_loss_std = [loss_maml0_std, 0.5712757040990449, 0.4035987596960742, 0.7908836370228796, 1.204131575401754, 1.4963189920140545, 0.8247595146034412, 0.4606890251519241, 1.104740023523924]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(inner_steps_for_dist, meta_test_cca, yerr=meta_test_cca_std, marker='x', label='dCCA')
axs[0].errorbar(inner_steps_for_dist, meta_test_ned, yerr=meta_test_ned_std, marker='x', label='NED')
axs[0].axhline(y=0.12, color='r', linestyle='--', label='dCCA previous work')
axs[0].legend()
axs[0].set_title('Representation difference vs adaption\'s inner steps ')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(inner_steps_for_loss, meta_test_loss, yerr=meta_test_loss_std, marker='x', label='loss', color='g')
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