#!/home/miranda9/.conda/envs/automl-meta-learning/bin/python3.7
#SBATCH --job-name="miranda9job"
#SBATCH --output="demo.%j.%N.out"
#SBATCH --error="demo.%j.%N.err"
#SBATCH --partition=secondary
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-user=brando.science@gmail.com
#SBATCH --mail-type=ALL

print('Hi')
