#!/homes/miranda9/.conda/envs/automl-meta-learning/bin/python
#PBS -V
#PBS -M brando.science@gmail.com
#PBS -m abe
#PBS -lselect=1:ncpus=112

#!/homes/miranda9/.conda/envs/automl-meta-learning/bin/python
#!/homes/miranda9/.conda/envs/automl-meta-learning/bin/python
#!/homes/miranda9/.conda/envs/myenv/lib/python3.7

import sys
import os

for p in sys.path:
    print(p)

print(os.environ)
