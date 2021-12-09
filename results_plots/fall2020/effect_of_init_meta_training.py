

#%%

# WIG: what is causing the high performance 79.64% of pff resnet12_rfs?
# question: is the meta-learning method the one causing it?
# plot 2 from notes
# method to test: fix everything and only varying the inits


#%%

# reference for bar MI_plots_sl_vs_maml_1st_attempt: https://benalexkeen.com/bar-charts-in-matplotlib/

# WIG: what is causing the high performances of pff?
# question: fix everything and only vary the inits with resnet_18 (pytorch)
# note: adaptaion = FF
# plot 1 from notes

init_2_meta_test_acc = {
    'not pt': 0.286,
    'pt': 0.5328,
    'SL 64': 0.5175,
    'MAML0': 0.23009999999999997,
    'MAML1 (15)': 0.5452000000000001,
    'MAML1 (100)': 0.5699000000000001,
    #'MAML4': 1.5
}

init_2_meta_test_acc_std = {
    'not pt': 0.00040000000000001146,
    'pt': 0.0088,
    'SL 64': 0.0073,
    'MAML0': 0.0013000000000000095,
    'MAML1 (15)': 0.008400000000000019,
    'MAML1 (100)': 0.014899999999999969,
    #'MAML4': 0.0
}

#%%

# reference for bar MI_plots_sl_vs_maml_1st_attempt: https://benalexkeen.com/bar-charts-in-matplotlib/

# WIG: what is causing the high performances of pff?
# question: fix everything and only vary the inits with resnet_18 (pytorch)
# note: adaptaion = MAML1 1e-1
# plot 1 from notes

init_2_meta_test_acc = {
    'not pt': 0.2801,
    'pt': 0.5328,
    'SL 64': 0.5175,
    'MAML0': 0.23009999999999997,
    'MAML1 (15)': 0.5452000000000001,
    'MAML1 (100)': 0.5699000000000001,
    #'MAML4': 1.5
}

init_2_meta_test_acc_std = {
    'not pt': 0.0111,
    'pt': 0.0088,
    'SL 64': 0.0073,
    'MAML0': 0.0013000000000000095,
    'MAML1 (15)': 0.008400000000000019,
    'MAML1 (100)': 0.014899999999999969,
    #'MAML4': 0.0
}

#%%

from pathlib import Path
import matplotlib.pyplot as plt

plt.style.use('ggplot')

root = Path('/').expanduser()

inits = list(init_2_meta_test_acc.keys())
meta_test_accs = list(init_2_meta_test_acc.values())
yerr = list(init_2_meta_test_acc_std.values())

x_pos = [i for i, _ in enumerate(inits)]

plt.bar(x_pos, meta_test_accs, color='green', yerr=yerr)
plt.xlabel("Initializations")
plt.ylabel("Meta-test Accuracy")
plt.title("Meta-test vs Meta-training Method")

plt.xticks(x_pos, inits)

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.tight_layout()

plt.savefig(root / 'effect_of_init_resnet18_pytorch.png')
plt.savefig(root / 'effect_of_init_resnet18_pytorch.svg')
plt.savefig(root / 'effect_of_init_resnet18_pytorch.pdf')

plt.show()

print('plotting!')


