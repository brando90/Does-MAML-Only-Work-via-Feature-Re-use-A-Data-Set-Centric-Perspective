#%%

# ml + loss vs inner steps (ReLU overfitting net)

import numpy as np
import matplotlib.pyplot as plt
from pylab import MaxNLocator

from pathlib import Path

print('running')

save_plot = True

inner_steps = [1, 2, 3, 4, 5]

loss_maml0 = 31.55276138186455
loss_maml0_std = 1.967432177299856

meta_test_loss = [18.441477685809136, 20.303904067516328, 25.32764632001519, 19.433446311354636, 18.490721735000612]
meta_test_loss_std = [1.635711418482081, 0.798269813924878, 3.4299102692349495, 3.0274966343723926, 2.1497806808846636]

meta_test_cca = [0.4586714406808217, 0.4457972725232443, 0.44969746967156726, 0.46067451536655424, 0.4610745032628378]
meta_test_cca_std = [0.010026041480742612, 0.03507832366557312, 0.016224168326099596, 0.010874420144336924, 0.034032159297672296]

meta_test_ned = [0.44390130274649825, 0.46010129586162574, 0.43565698151441634, 0.45785821755992695, 0.42849536148069517]
meta_test_ned_std = [0.010231025905967459, 0.017612652737178745, 0.016794130223352265, 0.009304949195193361, 0.016039790497987778]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(inner_steps, meta_test_cca, yerr=meta_test_cca_std, marker='o', label='cca')
axs[0].errorbar(inner_steps, meta_test_ned, yerr=meta_test_ned_std, marker='o', label='ned')
axs[0].legend()
axs[0].set_title('Rapid learning vs adaptation inner steps')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(inner_steps, meta_test_loss, yerr=meta_test_loss_std, marker='o', label='adapted')
axs[1].set_title('Meta-Validation loss vs adaptation inner steps')
axs[1].set_xlabel('adaptation\'s inner steps')
axs[1].set_ylabel('Loss')
axs[1].axhline(y=loss_maml0, color='r', linestyle='--', label='not adaptated')
axs[1].get_xaxis().set_major_locator(MaxNLocator(integer=True))

axs[1].legend()

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'ml_loss_vs_inner_steps.png')
    plt.savefig(root / 'ml_loss_vs_inner_steps.svg')
    plt.savefig(root / 'ml_loss_vs_inner_steps.pdf')

plt.show()

print('done')

#%%

# ml + loss vs inner steps (ReLU best net)

import numpy as np
import matplotlib.pyplot as plt
from pylab import MaxNLocator

from pathlib import Path

print('running')

save_plot = True
# save_plot = False

inner_steps = [1, 2, 3, 4, 5, 8, 16, 32]

loss_maml0 = 19.455371450126172
loss_maml0_std = 1.019144981585053

meta_test_loss = [8.909195451736451, 10.453863927662372, 10.438749479651452, 10.866869341522456, 10.897315060257913, 8.694210491478442, 9.191108769118786, 9.092517228752374]
meta_test_loss_std = [0.5712757040990449, 0.4035987596960742, 0.7908836370228796, 1.204131575401754, 1.4963189920140545, 0.8247595146034412, 0.4606890251519241, 1.104740023523924]

meta_test_cca = [0.5554760853449504, 0.5614992250998815, 0.5559805711110433, 0.5665635496377945, 0.5695259730021158, 0.5757676988840104, 0.5487342963616053, 0.5536342511574428]
meta_test_cca_std = [0.024697060891281774, 0.01635608540630873, 0.020506618487168466, 0.021074992715715904, 0.008751912905302727, 0.013803703536447796, 0.0040189992442360494, 0.01672345811738925]

meta_test_ned = [0.509952854503624, 0.49967966980659717, 0.5071488178037521, 0.509229507290115, 0.5086124367750696, 0.5105350886485593, 0.507246109916044, 0.5101748466011903]
meta_test_ned_std = [0.009346849133655227, 0.015704240939143948, 0.0024862043152066556, 0.009187025418748784, 0.016039790497987778, 0.0038458444722408187, 0.005801110523967555, 0.007980516656024433]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(inner_steps, meta_test_cca, yerr=meta_test_cca_std, marker='o', label='cca')
axs[0].errorbar(inner_steps, meta_test_ned, yerr=meta_test_ned_std, marker='o', label='ned')
axs[0].legend()
axs[0].set_title('Rapid learning vs adaptation inner steps')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(inner_steps, meta_test_loss, yerr=meta_test_loss_std, marker='o', label='adapted')
axs[1].set_title('Meta-Validation loss vs adaptation inner steps')
axs[1].set_xlabel('adaptation\'s inner steps')
axs[1].set_ylabel('Loss')
axs[1].axhline(y=loss_maml0, color='r', linestyle='--', label='not adaptated')
axs[1].get_xaxis().set_major_locator(MaxNLocator(integer=True))

axs[1].legend()

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'ml_loss_vs_inner_steps.png')
    plt.savefig(root / 'ml_loss_vs_inner_steps.svg')
    plt.savefig(root / 'ml_loss_vs_inner_steps.pdf')

plt.show()

print('done')

# %%

# ml + loss vs inner steps (Sigmoid Overfitting)

import numpy as np
import matplotlib.pyplot as plt
from pylab import MaxNLocator

from pathlib import Path

print('running')

save_plot = True
# save_plot = False

inner_steps = [1, 2, 4, 8, 16, 32]

loss_maml0 = 21.993277879297732
loss_maml0_std = 1.3118791154460914

meta_test_loss = [12.456569802880285, 12.300762733519075, 14.410906747341155, 11.86603080368042, 9.899388624727726, 10.151166315436361]
meta_test_loss_std = [1.0442358973321169, 0.2503485005144042, 1.5733223011681903, 0.651172950072793, 0.42793103283309547, 0.26417619153366617]

meta_test_cca = [0.5484926203886669, 0.563411878546079, 0.5710338006416956, 0.5666298776865005, 0.585236128171285, 0.5766304810841878]
meta_test_cca_std = [0.009700146965041796, 0.01871775899051635, 0.00971045966862034, 0.02078150453003017, 0.005158674528464366, 0.0026559672973737658]

meta_test_ned = [0.45667313226745126, 0.4162678280618136, 0.42653271106294455, 0.4306463763964289, 0.4276316414306406, 0.42947973532611605]
meta_test_ned_std = [0.010226682878174938, 0.0063777776948998185, 0.01761758802371641, 0.019391363481348442, 0.016798326983210016, 0.016331380119871584]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(inner_steps, meta_test_cca, yerr=meta_test_cca_std, marker='o', label='cca')
axs[0].errorbar(inner_steps, meta_test_ned, yerr=meta_test_ned_std, marker='o', label='ned')
axs[0].legend()
axs[0].set_title('Rapid learning vs adaptation inner steps')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(inner_steps, meta_test_loss, yerr=meta_test_loss_std, marker='o', label='adapted')
axs[1].set_title('Meta-Validation loss vs adaptation inner steps')
axs[1].set_xlabel('adaptation\'s inner steps')
axs[1].set_ylabel('Loss')
# axs[1].axhline(y=loss_maml0, color='r', linestyle='--', label='not adaptated')
axs[1].get_xaxis().set_major_locator(MaxNLocator(integer=True))

axs[1].legend()

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'ml_loss_vs_inner_steps.png')
    plt.savefig(root / 'ml_loss_vs_inner_steps.svg')
    plt.savefig(root / 'ml_loss_vs_inner_steps.pdf')

plt.show()

print('done')

# %%

# ml + loss vs inner steps (Sigmoid best val)

import numpy as np
import matplotlib.pyplot as plt
from pylab import MaxNLocator

from pathlib import Path

print('running')

save_plot = True
# save_plot = False

inner_steps = [1, 2, 4, 8, 16, 32]

loss_maml0 = 43.96553179168701
loss_maml0_std = 1.8058700680403383

meta_test_loss = [13.739946127653122, 8.793661404013632, 8.447031805932522, 8.42278252196312, 8.477399283349515, 8.312923258244993]
meta_test_loss_std = [1.0994678776141051, 0.36911269154459236, 0.432820357268575, 0.7749138218896927, 0.9523679944987157, 0.9205561753211284]

meta_test_cca = [0.5977237413326899, 0.5928615580002466, 0.5958830783764522, 0.598208557566007, 0.5973852763573329, 0.5957469274600347]
meta_test_cca_std = [0.00707934713573587, 0.00846237404064218, 0.007949951903377304, 0.0065036558630005474, 0.004912136217673985, 0.00909417287489179]

meta_test_ned = [0.5173536032550894, 0.5093832578420319, 0.511174541809931, 0.5154659108732735, 0.51203569236589, 0.5070959242477404]
meta_test_ned_std = [0.016695623433091045, 0.007882302117803776, 0.020698135909910757, 0.013762821990194829, 0.02757315414211051, 0.008426892297632705]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(inner_steps, meta_test_cca, yerr=meta_test_cca_std, marker='o', label='cca')
axs[0].errorbar(inner_steps, meta_test_ned, yerr=meta_test_ned_std, marker='o', label='ned')
axs[0].legend()
axs[0].set_title('Rapid learning vs adaptation inner steps')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(inner_steps, meta_test_loss, yerr=meta_test_loss_std, marker='o', label='adapted')
axs[1].set_title('Meta-Validation loss vs adaptation inner steps')
axs[1].set_xlabel('adaptation\'s inner steps')
axs[1].set_ylabel('Loss')
# axs[1].axhline(y=loss_maml0, color='r', linestyle='--', label='not adaptated')
axs[1].get_xaxis().set_major_locator(MaxNLocator(integer=True))

axs[1].legend()

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'ml_loss_vs_inner_steps.png')
    plt.savefig(root / 'ml_loss_vs_inner_steps.svg')
    plt.savefig(root / 'ml_loss_vs_inner_steps.pdf')

plt.show()
