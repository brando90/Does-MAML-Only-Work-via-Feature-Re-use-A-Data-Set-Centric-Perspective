"""
conda create -n uutils_env python=3.9
conda activate uutils_env
conda remove --all --name uutils_env
rm -rf /Users/brando/anaconda3/envs/uutils_env

pip install -e ~/ultimate-utils/ultimate-utils-proj-maml_vs_adapted_maml_src/

pip install ultimate-utils

To test it:
python -c "import uutils; uutils.hello()"
python -c "import uutils; uutils.torch_uu.hello()"

python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
python -c "import uutils; uutils.torch_uu.gpu_test()"

"""

from setuptools import setup
from setuptools import find_packages

import pathlib

# The directory containing this file
# HERE = pathlib.Path(__file__).parent
# HERE = pathlib.Path('~/ultimate-utils/').expanduser()
HERE = pathlib.Path('~/automl-meta-learning/').expanduser()

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='automl-meta-learning',  # project name
    version='0.1.0',
    description='Brandos automl project',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/brando90/automl-meta-learning',
    author='Brando Miranda',
    author_email='brandojazz@gmail.com',
    python_requires='>=3.9.0',
    license='MIT',
    packages=find_packages(),  # imports all modules (folder with __init__.py) & python files in this folder (since defualt args are . and empty exculde i.e. () )
    # basing the torch install from the pytorch website as of this writing: https://pytorch.org/get-started/locally/
    # pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    install_requires=[]  # left blant deliberately since vision cluster needs its own special install script
    # install_requires=['torch==1.9.1',
    #                   'torchvision==0.10.1',
    #                   'torchaudio==0.9.1',
    #                   'dill',
    #                   'networkx>=2.5',
    #                   'scipy',
    #                   'scikit-learn',
    #                   'lark-parser',
    #                   'torchtext==0.10.1',
    #                   'tensorboard',
    #                   'pandas',
    #                   'progressbar2',
    #                   'transformers',
    #                   'requests',
    #                   'aiohttp',
    #                   'numpy',
    #                   'plotly',
    #                   'wandb',
    #                   'matplotlib',
    #                   # 'seaborn'
    #
    #                   # 'pygraphviz'  # removing because it requires user to install graphviz and gives other issues, e.g. if the user does not want to do graph stuff then uutils shouldn't need to force the user to install uutils
    #                   ]
)
