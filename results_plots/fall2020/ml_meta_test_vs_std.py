#%%

# ml + loss vs std (overfitted)

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

print('running')

save_plot = False

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

axs[0].errorbar(stds, meta_test_cca, yerr=meta_test_cca_std, marker='o', label='cca')
axs[0].errorbar(stds, meta_test_ned, yerr=meta_test_ned_std, marker='o', label='ned')
axs[0].legend()
axs[0].set_title('Rapid learning vs std of data set')
# axs[0].set_xlabel('std of task')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(stds, meta_test_loss, yerr=meta_test_loss_std, marker='o', label='loss')
axs[1].legend()
axs[1].set_title('Meta-Validation loss vs std of task')
axs[1].set_xlabel('std of task')
axs[1].set_ylabel('Loss')

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'ml_loss_vs_inner_steps.png')
    plt.savefig(root / 'ml_loss_vs_inner_steps.svg')
    plt.savefig(root / 'ml_loss_vs_inner_steps.pdf')

plt.show()

print('done')

#%%

# ml + loss vs std (best)

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

print('running')

save_plot = False
save_plot = True

stds = [0.5, 1.0, 2.0, 4.0]

meta_test_loss = [7.625178372740746, 10.158726910948754, 8.668823894679546, 10.927158722639085]
meta_test_loss_std = [0.5123193061123283, 0.6722101351222379, 0.7487423250321502, 1.1042277611931715]

# these are integral difference
# meta_test_loss = [18.06054182490524, 21.024920472624352, 59.649208159349406, 53.42566085720061]
# meta_test_loss_std = [3.9390681461748627, 5.168308546143307, 4.093851180247897, 5.33135624648043]

meta_test_cca = [0.533019460241, 0.5275834341843922, 0.5326053589582443, 0.5199869662523269]
meta_test_cca_std = [0.031491566797677686, 0.019773647559281876, 0.009824007704873539, 0.015287316810794021]

meta_test_ned = [0.4536823472286004, 0.4746091389964411, 0.4771619125271688, 0.5002327398097794]
meta_test_ned_std = [0.005222276584721787, 0.008115222132595508, 0.004904429800945309, 0.012230113038618864]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(stds, meta_test_cca, yerr=meta_test_cca_std, marker='o', label='cca')
axs[0].errorbar(stds, meta_test_ned, yerr=meta_test_ned_std, marker='o', label='ned')
axs[0].legend()
axs[0].set_title('Rapid learning vs std of data set')
# axs[0].set_xlabel('std of task')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(stds, meta_test_loss, yerr=meta_test_loss_std, marker='o', label='loss')
axs[1].legend()
axs[1].set_title('Meta-Validation loss vs std of task')
axs[1].set_xlabel('std of task')
axs[1].set_ylabel('Loss')

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'ml_loss_vs_inner_steps.png')
    plt.savefig(root / 'ml_loss_vs_inner_steps.svg')
    plt.savefig(root / 'ml_loss_vs_inner_steps.pdf')

plt.show()

print('done')

#%%

# ml + loss vs std (overfitted sigmoid)

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

print('running')

save_plot = False
save_plot = True

stds = [0.5, 1.0, 2.0]

meta_test_loss = [10.580027156233788, 13.139623494029047, 13.294900600194932]
meta_test_loss_std = [0.5950659424541982, 0.6898089007537416, 0.9440124825108793]

# these are integral difference
# meta_test_loss = [18.06054182490524, 21.024920472624352, 59.649208159349406, 53.42566085720061]
# meta_test_loss_std = [3.9390681461748627, 5.168308546143307, 4.093851180247897, 5.33135624648043]

meta_test_cca = [0.5164651145537694, 0.5627435942490896, 0.5405335744222006]
meta_test_cca_std = [0.010537784400415737, 0.009425336097504615, 0.012497050542139785]

meta_test_ned = [0.5003526682888336, 0.39112880632976266, 0.4138465576192004]
meta_test_ned_std = [0.004511986599098143, 0.013464183816837231, 0.010426005650793508]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(stds, meta_test_cca, yerr=meta_test_cca_std, marker='o', label='cca')
axs[0].errorbar(stds, meta_test_ned, yerr=meta_test_ned_std, marker='o', label='ned')
axs[0].legend()
axs[0].set_title('Rapid learning vs std of data set')
# axs[0].set_xlabel('std of task')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(stds, meta_test_loss, yerr=meta_test_loss_std, marker='o', label='loss')
axs[1].legend()
axs[1].set_title('Meta-Validation loss vs std of task')
axs[1].set_xlabel('std of task')
axs[1].set_ylabel('Loss')

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'ml_loss_vs_inner_steps.png')
    plt.savefig(root / 'ml_loss_vs_inner_steps.svg')
    plt.savefig(root / 'ml_loss_vs_inner_steps.pdf')

plt.show()

print('done')

#%%

# ml + loss vs std (best sigmoid)

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

print('running')

save_plot = False
save_plot = True

stds = [0.5, 1.0, 2.0]

meta_test_loss = [8.411029676139353, 9.56819876730442, 8.450394897818565]
meta_test_loss_std = [1.1997668672519493, 0.49257183880098115, 0.8088200923741082]

# these are integral difference
# meta_test_loss = [18.06054182490524, 21.024920472624352, 59.649208159349406, 53.42566085720061]
# meta_test_loss_std = [3.9390681461748627, 5.168308546143307, 4.093851180247897, 5.33135624648043]

meta_test_cca = [0.6099943568309147, 0.591276615858078, 0.5984459360440572]
meta_test_cca_std = [0.004636955227907892, 0.011655829472283151, 0.012917641760539672]

meta_test_ned = [0.4966479776368378, 0.39724864388814873, 0.5129354873081231]
meta_test_ned_std = [0.007917491972751553, 0.02717265626316231, 0.0072700336332698675]

# plt.title("Meta-test vs Depth of ResNet")

fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)

axs[0].errorbar(stds, meta_test_cca, yerr=meta_test_cca_std, marker='o', label='cca')
axs[0].errorbar(stds, meta_test_ned, yerr=meta_test_ned_std, marker='o', label='ned')
axs[0].legend()
axs[0].set_title('Rapid learning vs std of data set')
# axs[0].set_xlabel('std of task')
axs[0].set_ylabel('Represenation change')
# axs[0].set_ylim([0, 1])

axs[1].errorbar(stds, meta_test_loss, yerr=meta_test_loss_std, marker='o', label='loss')
axs[1].legend()
axs[1].set_title('Meta-Validation loss vs std of task')
axs[1].set_xlabel('std of task')
axs[1].set_ylabel('Loss')

plt.tight_layout()

if save_plot:
    root = Path('~/Desktop').expanduser()
    plt.savefig(root / 'ml_loss_vs_inner_steps.png')
    plt.savefig(root / 'ml_loss_vs_inner_steps.svg')
    plt.savefig(root / 'ml_loss_vs_inner_steps.pdf')

plt.show()

print('done')

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
