import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# %%

root = Path('/').expanduser()

bb = [18, 50, 101, 152]

meta_test_acc = [0.653, 0.366, 0.387, 0.177]
meta_test_acc_std = [0.0400, 0.0200, 0.0400, 0.00333]

plt.errorbar(bb, meta_test_acc, yerr=meta_test_acc_std)
plt.xlabel("Depth of ResNet")
plt.ylabel("Meta-test Accuracy")
plt.title("Meta-test vs Depth of ResNet")

plt.tight_layout()

plt.savefig(root / 'effect_of_bb_resnet18_pytorch.png')
plt.savefig(root / 'effect_of_bb_resnet18_pytorch.svg')
plt.savefig(root / 'effect_of_bb_resnet18_pytorch.pdf')

root = Path('/').expanduser()