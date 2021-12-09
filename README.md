# Does-MAML-Only-Work-via-Feature-Re-use-A-Data-Set-Centric-Perspective
Does MAML Only Work via Feature Re-use? A Data Set Centric Perspective

# Installing

## Standard pip instal [Recommended]

TODO

If you are going to use a gpu the do this first before continuing 
(or check the offical website: https://pytorch.org/get-started/locally/):
```angular2html
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Otherwise, just doing the follwoing should work.
```
pip install automl
```
If that worked, then you should be able to import is as follows:
```
import automl
```

## Manual installation [Development]

To use library first get the code from this repo (e.g. fork it on github):

```
git clone git@github.com/brando90/automl-meta-learning.git
```

Then install it in development mode in your python env with python >=3.9
(read `modules_in_python.md` to learn about python envs in uutils).
E.g. create your env with conda:

```
conda create -n metalearning python=3.9
conda activate metalearning
```

Then install it in edibable mode and all it's depedencies with pip in the currently activated conda environment:

```
pip install -e ~/automl-meta-learning/automl-proj-src/
```

since the depedencies have not been written install them:

```
pip install -e ~/ultimate-utils/ultimate-utils-proj-src
```

then test as followsing:
```
python -c "import uutils; print(uutils); uutils.hello()"
python -c "import meta_learning; print(meta_learning)"
python -c "import meta_learning; print(meta_learning); meta_learning.hello()"
```
output should be something like this:
```
(metalearning) brando~/automl-meta-learning/automl-proj-src ❯ python -c "import uutils; print(uutils); uutils.hello()"
<module 'uutils' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/__init__.py'>

hello from uutils __init__.py in:
<module 'uutils' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/__init__.py'>

(metalearning) brando~/automl-meta-learning/automl-proj-src ❯ python -c "import meta_learning; print(meta_learning)"
<module 'meta_learning' from '/Users/brando/automl-meta-learning/automl-proj-src/meta_learning/__init__.py'>
(metalearning) brando~/automl-meta-learning/automl-proj-src ❯ python -c "import meta_learning; print(meta_learning); meta_learning.hello()"
<module 'meta_learning' from '/Users/brando/automl-meta-learning/automl-proj-src/meta_learning/__init__.py'>

hello from torch_uu __init__.py in:
<module 'uutils.torch_uu' from '/Users/brando/ultimate-utils/ultimate-utils-proj-src/uutils/torch_uu/__init__.py'>

```

# Reproducing Results

TODO

## Citation

```
B. Miranda, Y.Wang, O. Koyejo.
Does MAML Only Work via Feature Re-use? A Data Set Centric Perspective. 
(Planned Release Date December 2021).
https://drive.google.com/file/d/1cTrfh-Tg39EnbI7u0-T29syyDp6e_gjN/view?usp=sharing
```

https://drive.google.com/file/d/1cTrfh-Tg39EnbI7u0-T29syyDp6e_gjN/view?usp=sharing
